#!/usr/bin/env python3
"""Write or check the expectations of conformance tests, with mmc as oracle.

A test in test/Conformance (docs/INPROGRESS.md, A conformance suite) is a
Mercury module whose header names what it tests (`% MANUAL:` or `% DESIGN:`)
and whose class is the start of its file name:

  run-      compiles, runs, and prints what mmc's program prints, with mmc's
            exit status;
  reject-   is rejected, with an error at each line marked `% ERROR`, which
            are the lines mmc reports errors at;
  nyi-      gives merkc's "not yet implemented" error; mmc must compile and
            run it, so that it becomes a run- test when the feature lands;
  differs-  prints what merkc's program prints, where it differs from mmc's
            for the reason its `% DIFFERS:` line gives; mmc's output is kept
            in `% MMC:` lines, so that a change in either shows. Where mmc
            rejects the program, as for a limitation of its implementation,
            `% MMC-STATUS: rejected` and its errors' first lines are kept.

A run- or reject- test that merkc gets wrong, a known bug rather than a
feature not done, says why in a `% KNOWN-BUG:` line: its expectations are
still mmc's, and the script adds lit's `% XFAIL: *`, so that the test is
expected to fail until the bug is fixed, and fails (XPASS) once it is, until
the line goes. matrix.py --known lists them.

The script owns the header's lines that start `% RUN:`, `% OUTPUT`,
`% ERRORS`, `% NYI`, `% MMC`, `% STATUS` and `% XFAIL:`, and writes them at the
end of the header. Other header lines are the author's:

  % MANUAL: <a node of the reference manual, as matrix.py lists them>
  % DESIGN: <a section of the design document>
  % MODULES: Inputs/other.m ...   more modules of the program
  % FILES: Inputs/facts ...       data files the program's build reads, such
                                  as a fact table's, copied beside it for
                                  mmc but not compiled
  % MERKC-FLAGS: ...              more flags for merkc; with -fopenmp,
                                  the program is linked with libomp and
                                  run on four threads, and also built
                                  without -fopenmp, and must print the
                                  same both ways
  % FLAGS-FOR-MMC: ...            more flags for mmc, as for --trace-flag
  % DIFFERS: <mmc-bug|manual|implementation>: <why>
  % KNOWN-BUG: <what merkc gets wrong, and where it is recorded>
  % TRANSLATION: Inputs/translation.m
                                  the module mmc builds and runs in place of
                                  the test, for an extension mmc does not
                                  have: a loop's translation written by hand
                                  as plain Mercury, whose output the test's
                                  must be; or Inputs/translation.pl, a
                                  program that SWI-Prolog runs, main/0 in
                                  the file's directory, for open terms
  % ORACLE: none                  expectations written by hand, which the
                                  script leaves as they are, for a test of an
                                  extension that has no translation, as an
                                  error only merkc reports, or a warn- test

By default it checks that each test's lines are what it would write, and
exits with 1 if any are not; with --update it writes them. With --triage it
also builds each run- test with merkc and says whether merkc's program
agrees with mmc's, for the triage of a difference: our bug, an mmc bug, or
behaviour the implementation chooses.

Usage: oracle.py [--update] [--triage] [--mmc MMC] [--merkc MERKC]
                 [--jobs N] PATH...
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

GENERATED = ("% RUN:", "% OUTPUT", "% ERRORS", "% NYI", "% MMC", "% STATUS",
             "% XFAIL:", "% REQUIRES:")
CLASSES = ("run", "reject", "nyi", "differs")
DIFFERS_KINDS = ("mmc-bug", "manual", "implementation")
FILECHECK_OUTPUT = (
    "FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace")


class TestError(Exception):
    pass


def run(command, cwd, timeout=120, merge=False, env=None):
    try:
        result = subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout, env=env) if not merge else subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            env=env)
    except subprocess.TimeoutExpired:
        raise TestError("timed out: " + " ".join(command))
    out = result.stdout.decode("utf-8", "replace")
    err = "" if merge else result.stderr.decode("utf-8", "replace")
    return result.returncode, out, err


class Test:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.kind = self.name.split("-", 1)[0]
        with open(path, encoding="utf-8") as f:
            self.lines = f.read().split("\n")
        # A test whose expectations are written by hand, such as a warn-
        # test of an extension, may be of a class the script does not know.
        if self.kind not in CLASSES and not any(
                re.match(r"% ORACLE:\s*none\s*$", l) for l in self.lines):
            raise TestError("the name starts with none of " +
                            ", ".join(c + "-" for c in CLASSES))
        self.header_end = 0
        while (self.header_end < len(self.lines) and
               self.lines[self.header_end].startswith("%")):
            self.header_end += 1
        header = self.lines[:self.header_end]
        self.authored = [l for l in header if not l.startswith(GENERATED)]
        # The `%' line that sets the generated lines off is the script's.
        while self.authored and self.authored[-1].strip() == "%":
            self.authored.pop()
        self.body = self.lines[self.header_end:]
        self.module = None
        for line in self.body:
            m = re.match(r":- module ([a-z][A-Za-z0-9_]*)\.", line)
            if m:
                self.module = m.group(1)
                break
        if not self.module:
            raise TestError("no `:- module' declaration")
        self.modules = []
        self.files = []
        self.flags = []
        self.mmc_flags = []
        self.differs = None
        self.known_bug = None
        self.mmc_program = None
        self.no_oracle = False
        for line in self.authored:
            m = re.match(r"% MODULES:(.*)", line)
            if m:
                self.modules += m.group(1).split()
            m = re.match(r"% FILES:(.*)", line)
            if m:
                self.files += m.group(1).split()
            m = re.match(r"% MERKC-FLAGS:(.*)", line)
            if m:
                self.flags += m.group(1).split()
            m = re.match(r"% FLAGS-FOR-MMC:(.*)", line)
            if m:
                self.mmc_flags += m.group(1).split()
            m = re.match(r"% DIFFERS: *([a-z-]+):", line)
            if m:
                self.differs = m.group(1)
            m = re.match(r"% KNOWN-BUG:\s*(\S.*)", line)
            if m:
                self.known_bug = m.group(1)
            m = re.match(r"% TRANSLATION:\s*(\S+)", line)
            if m:
                self.mmc_program = m.group(1)
            if re.match(r"% ORACLE:\s*none\s*$", line):
                self.no_oracle = True
        if not any(l.startswith(("% MANUAL:", "% DESIGN:"))
                   for l in self.authored):
            raise TestError("no `% MANUAL:' or `% DESIGN:' line")
        if self.kind == "differs" and self.differs not in DIFFERS_KINDS:
            raise TestError("a differs- test needs `% DIFFERS: KIND: why', "
                            "KIND one of " + ", ".join(DIFFERS_KINDS))
        if self.mmc_program and self.kind not in ("run", "nyi"):
            raise TestError("only a run- or nyi- test has a TRANSLATION")
        if self.known_bug and self.kind not in ("run", "reject"):
            raise TestError("only a run- or reject- test has a known bug; "
                            "a %s- test says what merkc does" % self.kind)

    def openmp(self):
        """Whether merkc compiles the program with -fopenmp: then it is run
        on four threads, and also built without -fopenmp, and prints the
        same either way."""
        return "-fopenmp" in self.flags

    def module_paths(self):
        here = os.path.dirname(self.path)
        return [os.path.join(here, m) for m in self.modules]

    def copy_to(self, directory):
        """The program's modules in a directory, as mmc --make wants them,
        and the module to make: the test's, or its TRANSLATION's."""
        here = os.path.dirname(self.path)
        module = self.module
        if self.mmc_program:
            program = os.path.join(here, self.mmc_program)
            with open(program, encoding="utf-8") as f:
                m = re.search(r"^:- module ([a-z][A-Za-z0-9_]*)\.", f.read(),
                              re.M)
            if not m:
                raise TestError("no `:- module' declaration in " +
                                self.mmc_program)
            module = m.group(1)
            shutil.copy(program, os.path.join(directory, module + ".m"))
        else:
            shutil.copy(self.path, os.path.join(directory, self.module + ".m"))
        for path in self.module_paths() + [os.path.join(here, f)
                                           for f in self.files]:
            shutil.copy(path, directory)
        return module

    def marked_lines(self, generated):
        """The lines marked `% ERROR`, numbered as in the file with the
        generated lines in its header."""
        first = len(self.authored) + 1 + len(generated) + 1
        return [first + i for i, line in enumerate(self.body)
                if re.search(r"%\s*ERROR\s*$", line)]

    def text(self, generated):
        return "\n".join(self.authored + ["%"] + generated + self.body)


def mmc_run(test, mmc):
    """mmc's program's output, standard error after standard output as
    they are written, and its exit status; or, for a translation into
    Prolog, swipl's, which runs its main/0 in the translation's directory,
    where the helpers it loads are."""
    if test.mmc_program and test.mmc_program.endswith(".pl"):
        program = os.path.join(os.path.dirname(test.path), test.mmc_program)
        status, out, _ = run(["swipl", "-q", "-g", "main", "-t", "halt",
                              os.path.basename(program)],
                             os.path.dirname(program), merge=True)
        return status, out
    with tempfile.TemporaryDirectory() as tmp:
        module = test.copy_to(tmp)
        status, out, err = run([mmc] + test.mmc_flags + ["--make", module],
                               tmp, timeout=600)
        if status != 0:
            raise TestError("mmc cannot compile it:\n" + out + err)
        status, out, _ = run(["./" + module], tmp, merge=True)
        return status, out


def mmc_errors(test, mmc):
    """The lines of the test that mmc reports errors at, if it rejects it."""
    return {line for line, _ in mmc_error_texts(test, mmc)}


def mmc_error_texts(test, mmc):
    """The line and text of each error mmc reports in the test, if it
    rejects it, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        test.copy_to(tmp)
        # --make builds the interfaces of the modules it imports first.
        status, out, err = run(
            [mmc] + test.mmc_flags + ["--make", test.module + ".err"], tmp,
            timeout=600)
        if status == 0:
            raise TestError("mmc accepts it")
        # mmc --make cuts what it writes short ("error log truncated");
        # the module's .err file has all of it, unless the errors were
        # found while making its interface files, when it is empty.
        log = out + err
        path = os.path.join(tmp, test.module + ".err")
        if os.path.exists(path):
            with open(path) as f:
                log = f.read() or log
        messages = []
        for line in log.split("\n"):
            m = re.match(r"(\S+?\.m):(\d+): (.*)", line)
            if not m or m.group(1) != test.module + ".m":
                continue
            if not m.group(3).startswith(" ") or not messages:
                messages.append((int(m.group(2)), []))
            messages[-1][1].append(m.group(3))
        errors = []
        for number, text in messages:
            joined = " ".join(text)
            # mmc writes "Error:" and "Warning:"; a message that is
            # neither, as "In clause for ...", leads to an error.
            # A note that points at the declaration an error conflicts
            # with is not an error of its own.
            if re.match(r"(Previous .* declaration was here|"
                        r"Here is the conflicting .*|"
                        r"The first one was here)\.$", joined):
                continue
            if re.search(r"\berror\b", joined, re.I) or not re.search(
                    r"\bwarning\b", joined, re.I):
                errors.append((number, " ".join(t.strip() for t in text)))
        return errors


def merkc_arguments(test, lit):
    """merkc's flags, search directories and sources, as a RUN line writes
    them, or as paths for the triage."""
    here = "%S" if lit else os.path.dirname(test.path)
    directories = sorted({os.path.dirname(m) for m in test.modules})
    return (test.flags +
            [a for d in directories for a in ("-I", here + "/" + d if d
                                               else here)] +
            ["%s" if lit else test.path] +
            [here + "/" + m for m in test.modules])


def libomp_dirs(merkc):
    """The directories of libomp and libomptarget in merkc's build, as the
    lit configuration finds them."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(merkc)))
    lib = os.path.join(root, "lib")
    candidates = [lib] + ([os.path.join(lib, d) for d in sorted(os.listdir(lib))
                           if "-linux" in d] if os.path.isdir(lib) else [])
    candidates.append(os.path.join(root, "runtimes", "runtimes-bins",
                                   "openmp", "runtime", "src"))
    dirs = []
    for library in ("libomp.so", "libomptarget.so"):
        found = [d for d in candidates
                 if os.path.exists(os.path.join(d, library))]
        if not found:
            raise TestError("no %s in merkc's build" % library)
        if found[0] not in dirs:
            dirs.append(found[0])
    return dirs


def merkc_build(test, merkc, level, directory, sequential=False):
    """Builds the program; an OpenMP test's with libomp, or, `sequential`,
    without -fopenmp."""
    exe = os.path.join(directory, "prog%d%s" % (level, "s" if sequential
                                                 else ""))
    arguments = merkc_arguments(test, False)
    if test.openmp():
        if sequential:
            arguments = [a for a in arguments if a != "-fopenmp"]
        else:
            arguments += ["-L" + d for d in libomp_dirs(merkc)]
    status, out, err = run([merkc, "-O%d" % level] + arguments + ["-o", exe],
                           directory, timeout=600)
    return status, out + err, exe


def run_program(test, merkc, exe, directory):
    """Runs a built program; an OpenMP test's on four threads."""
    env = None
    if test.openmp():
        env = dict(os.environ, OMP_NUM_THREADS="4",
                   LD_LIBRARY_PATH=":".join(libomp_dirs(merkc)))
    return run([exe], directory, merge=True, env=env)


def escape(line):
    """A line of output as a FileCheck pattern that matches it exactly."""
    result = ""
    i = 0
    while i < len(line):
        if line.startswith("{{", i):
            result += "{{[{][{]}}"
            i += 2
        elif line.startswith("[[", i):
            result += "{{\\[\\[}}"
            i += 2
        else:
            result += line[i]
            i += 1
    stripped = result.rstrip(" \t")
    if stripped != result:
        result = stripped + "{{" + result[len(stripped):] + "}}"
    return result


def output_lines(prefix, output):
    lines = output.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    checks = []
    for i, line in enumerate(lines):
        if i == 0:
            checks.append("%% %s:%s" % (prefix, escape(line) if line
                                         else "{{^$}}"))
        elif line == "":
            checks.append("%% %s-EMPTY:" % prefix)
        else:
            checks.append("%% %s-NEXT:%s" % (prefix, escape(line)))
    checks.append("%% %s-NOT:{{.}}" % prefix)
    return checks


def run_lines(test, status):
    arguments = " ".join(merkc_arguments(test, True))
    expect = "not " if status != 0 else ""
    if test.openmp():
        # Run on four threads, and built without -fopenmp, the program
        # prints the same.
        sequential = " ".join(a for a in merkc_arguments(test, True)
                              if a != "-fopenmp")
        threads = "%libomp-env OMP_NUM_THREADS=4 "
        return [
            "% REQUIRES: libomp",
            "% RUN: merkc -O0 " + arguments + " %libomp-flags -o %t0",
            "% RUN: " + threads + expect + "%t0 2>&1 | " + FILECHECK_OUTPUT,
            "% RUN: " + threads + "MERK_GC_EVERY_ALLOC=1 " + expect +
            "%t0 2>&1 | " + FILECHECK_OUTPUT,
            "% RUN: merkc -O2 " + arguments + " %libomp-flags -o %t2",
            "% RUN: " + threads + expect + "%t2 2>&1 | " + FILECHECK_OUTPUT,
            "% RUN: merkc -O0 " + sequential + " -o %t0s",
            "% RUN: " + expect + "%t0s 2>&1 | " + FILECHECK_OUTPUT,
            "% RUN: merkc -O2 " + sequential + " -o %t2s",
            "% RUN: " + expect + "%t2s 2>&1 | " + FILECHECK_OUTPUT,
        ]
    return [
        "% RUN: merkc -O0 " + arguments + " -o %t0",
        "% RUN: " + expect + "%t0 2>&1 | " + FILECHECK_OUTPUT,
        "% RUN: env MERK_GC_EVERY_ALLOC=1 " + expect + "%t0 2>&1 | " +
        FILECHECK_OUTPUT,
        "% RUN: merkc -O2 " + arguments + " -o %t2",
        "% RUN: " + expect + "%t2 2>&1 | " + FILECHECK_OUTPUT,
    ]


def generate(test, args):
    """The test's generated lines, and notes for the triage."""
    notes = []
    if test.kind == "run":
        status, out = mmc_run(test, args.mmc)
        generated = run_lines(test, status) + output_lines("OUTPUT", out)
        if args.triage:
            notes += triage(test, args.merkc, status, out)
        return generated, notes
    if test.kind == "reject":
        lines = mmc_errors(test, args.mmc)
        generated = [
            "% RUN: not merkc " + " ".join(merkc_arguments(test, True)) +
            " -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS "
            "--implicit-check-not=error:"]
        # One ERRORS line for each marked line, which the numbers count.
        markers = len(test.marked_lines([]))
        marked = test.marked_lines(generated + [""] * markers)
        # mmc numbers the lines of the file as it is.
        as_is = [i + 1 for i, line in enumerate(test.lines)
                 if i >= test.header_end and re.search(r"%\s*ERROR\s*$", line)]
        if set(as_is) != lines:
            raise TestError("mmc reports errors at lines %s, and the test "
                            "marks %s" % (sorted(lines), sorted(as_is)))
        generated += ["%% ERRORS-DAG: {{.*}}:%d:{{[0-9]+}}: error:" % n
                      for n in marked]
        return generated, notes
    if test.kind == "nyi":
        mmc_run(test, args.mmc)
        with tempfile.TemporaryDirectory() as tmp:
            status, out, _ = merkc_build(test, args.merkc, 0, tmp)
        m = re.search(r"error: (not yet implemented: .*)", out)
        if status == 0 or not m:
            raise TestError("merkc gives no `not yet implemented' error; "
                            "make it a run- test")
        return [
            "% RUN: not merkc " + " ".join(merkc_arguments(test, True)) +
            " -o %t 2>&1 | FileCheck %s --check-prefix=NYI",
            "% NYI: {{.*}}: error: " + escape(m.group(1))], notes
    # differs
    try:
        mmc_status, mmc_out = mmc_run(test, args.mmc)
    except TestError:
        mmc_status, mmc_out = None, "\n".join(
            "%d: %s" % (line, text) for line, text in
            mmc_error_texts(test, args.mmc)) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        status, out, exe = merkc_build(test, args.merkc, 0, tmp)
        if status != 0:
            raise TestError("merkc cannot compile it:\n" + out)
        status, out, _ = run_program(test, args.merkc, exe, tmp)
    if mmc_status is not None and (status != 0) == (mmc_status != 0) and \
            out == mmc_out:
        raise TestError("merkc's program agrees with mmc's; make it a run- "
                        "test")
    generated = run_lines(test, status) + output_lines("OUTPUT", out)
    generated.append("% MMC-STATUS: " + (
        "rejected" if mmc_status is None else "%d" % mmc_status))
    generated += [("% MMC: " + l).rstrip() for l in
                  mmc_out.rstrip("\n").split("\n")]
    return generated, notes


def triage(test, merkc, mmc_status, mmc_out):
    notes = []
    for sequential in ([False, True] if test.openmp() else [False]):
        which = "merkc" + (" without -fopenmp" if sequential else "")
        with tempfile.TemporaryDirectory() as tmp:
            status, out, exe = merkc_build(test, merkc, 0, tmp, sequential)
            if status != 0:
                notes.append(which + " cannot compile it: " +
                             (out.strip().split("\n") or [""])[0])
                continue
            status, out, _ = run_program(test, merkc, exe, tmp)
        if (status != 0) != (mmc_status != 0):
            notes.append("%s: exit status %d, mmc's %d" %
                         (which, status, mmc_status))
        if out != mmc_out:
            import difflib
            notes.append(which + ": output differs from mmc's:\n" +
                         "".join(difflib.unified_diff(
                             mmc_out.splitlines(True), out.splitlines(True),
                             "mmc", which)))
    return notes


def process(path, args):
    try:
        test = Test(path)
        if test.no_oracle:
            return path, "ok", []
        generated, notes = generate(test, args)
        if test.known_bug:
            generated.insert(0, "% XFAIL: *")
    except TestError as e:
        return path, "error", [str(e)]
    if test.text(generated) == "\n".join(test.lines):
        return path, "ok", notes
    if args.update:
        with open(path, "w", encoding="utf-8") as f:
            f.write(test.text(generated))
        return path, "updated", notes
    return path, "stale", notes


def tests_in(paths):
    for path in paths:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != "Inputs")
                for name in sorted(files):
                    if name.endswith(".m"):
                        yield os.path.join(root, name)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--triage", action="store_true")
    parser.add_argument("--mmc", default=os.environ.get("MMC") or
                        shutil.which("mmc"))
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--merkc", default=os.path.join(
        here, "..", "..", "..", "..", "build-merk", "bin", "merkc"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() // 2)
    args = parser.parse_args()
    if not args.mmc:
        sys.exit("oracle.py: no mmc: give --mmc, or put it on PATH")
    args.merkc = os.path.abspath(args.merkc)
    failed = False
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        for path, verdict, notes in pool.map(
                lambda p: process(p, args), list(tests_in(args.paths))):
            if verdict != "ok" or notes:
                print("%s: %s" % (verdict, path))
                for note in notes:
                    print("  " + note.replace("\n", "\n  ").rstrip())
            failed |= verdict in ("error", "stale") or bool(notes)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
