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
            in `% MMC:` lines, so that a change in either shows.

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
  % MERKC-FLAGS: ...              more flags for merkc
  % DIFFERS: <mmc-bug|manual|implementation>: <why>
  % KNOWN-BUG: <what merkc gets wrong, and where it is recorded>

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
             "% XFAIL:")
CLASSES = ("run", "reject", "nyi", "differs")
DIFFERS_KINDS = ("mmc-bug", "manual", "implementation")
FILECHECK_OUTPUT = (
    "FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace")


class TestError(Exception):
    pass


def run(command, cwd, timeout=120, merge=False):
    try:
        result = subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout) if not merge else subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
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
        if self.kind not in CLASSES:
            raise TestError("the name starts with none of " +
                            ", ".join(c + "-" for c in CLASSES))
        with open(path, encoding="utf-8") as f:
            self.lines = f.read().split("\n")
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
        self.flags = []
        self.differs = None
        self.known_bug = None
        for line in self.authored:
            m = re.match(r"% MODULES:(.*)", line)
            if m:
                self.modules += m.group(1).split()
            m = re.match(r"% MERKC-FLAGS:(.*)", line)
            if m:
                self.flags += m.group(1).split()
            m = re.match(r"% DIFFERS: *([a-z-]+):", line)
            if m:
                self.differs = m.group(1)
            m = re.match(r"% KNOWN-BUG:\s*(\S.*)", line)
            if m:
                self.known_bug = m.group(1)
        if not any(l.startswith(("% MANUAL:", "% DESIGN:"))
                   for l in self.authored):
            raise TestError("no `% MANUAL:' or `% DESIGN:' line")
        if self.kind == "differs" and self.differs not in DIFFERS_KINDS:
            raise TestError("a differs- test needs `% DIFFERS: KIND: why', "
                            "KIND one of " + ", ".join(DIFFERS_KINDS))
        if self.known_bug and self.kind not in ("run", "reject"):
            raise TestError("only a run- or reject- test has a known bug; "
                            "a %s- test says what merkc does" % self.kind)

    def module_paths(self):
        here = os.path.dirname(self.path)
        return [os.path.join(here, m) for m in self.modules]

    def copy_to(self, directory):
        """The program's modules in a directory, as mmc --make wants them."""
        shutil.copy(self.path, os.path.join(directory, self.module + ".m"))
        for path in self.module_paths():
            shutil.copy(path, directory)

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
    they are written, and its exit status."""
    with tempfile.TemporaryDirectory() as tmp:
        test.copy_to(tmp)
        status, out, err = run([mmc, "--make", test.module], tmp, timeout=600)
        if status != 0:
            raise TestError("mmc cannot compile it:\n" + out + err)
        status, out, _ = run(["./" + test.module], tmp, merge=True)
        return status, out


def mmc_errors(test, mmc):
    """The lines of the test that mmc reports errors at, if it rejects it."""
    with tempfile.TemporaryDirectory() as tmp:
        test.copy_to(tmp)
        # --make builds the interfaces of the modules it imports first.
        status, out, err = run(
            [mmc, "--make", test.module + ".err"], tmp, timeout=600)
        if status == 0:
            raise TestError("mmc accepts it")
        messages = []
        for line in (out + err).split("\n"):
            m = re.match(r"(\S+?\.m):(\d+): (.*)", line)
            if not m or m.group(1) != test.module + ".m":
                continue
            if not m.group(3).startswith(" ") or not messages:
                messages.append((int(m.group(2)), []))
            messages[-1][1].append(m.group(3))
        lines = set()
        for number, text in messages:
            joined = " ".join(text)
            if re.search(r"\berror\b", joined) or not re.search(
                    r"\bwarning\b", joined):
                lines.add(number)
        return lines


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


def merkc_build(test, merkc, level, directory):
    exe = os.path.join(directory, "prog%d" % level)
    status, out, err = run([merkc, "-O%d" % level] +
                           merkc_arguments(test, False) + ["-o", exe],
                           directory, timeout=600)
    return status, out + err, exe


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
    mmc_status, mmc_out = mmc_run(test, args.mmc)
    with tempfile.TemporaryDirectory() as tmp:
        status, out, exe = merkc_build(test, args.merkc, 0, tmp)
        if status != 0:
            raise TestError("merkc cannot compile it:\n" + out)
        status, out, _ = run([exe], tmp, merge=True)
    if (status != 0) == (mmc_status != 0) and out == mmc_out:
        raise TestError("merkc's program agrees with mmc's; make it a run- "
                        "test")
    generated = run_lines(test, status) + output_lines("OUTPUT", out)
    generated.append("% MMC-STATUS: %d" % mmc_status)
    generated += ["% MMC: " + l for l in mmc_out.rstrip("\n").split("\n")]
    return generated, notes


def triage(test, merkc, mmc_status, mmc_out):
    with tempfile.TemporaryDirectory() as tmp:
        status, out, exe = merkc_build(test, merkc, 0, tmp)
        if status != 0:
            return ["merkc cannot compile it: " +
                    (out.strip().split("\n") or [""])[0]]
        status, out, _ = run([exe], tmp, merge=True)
    notes = []
    if (status != 0) != (mmc_status != 0):
        notes.append("exit status %d, mmc's %d" % (status, mmc_status))
    if out != mmc_out:
        import difflib
        notes.append("output differs from mmc's:\n" + "".join(
            difflib.unified_diff(mmc_out.splitlines(True),
                                 out.splitlines(True), "mmc", "merkc")))
    return notes


def process(path, args):
    try:
        test = Test(path)
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
