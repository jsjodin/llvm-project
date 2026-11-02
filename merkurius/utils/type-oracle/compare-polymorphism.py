#!/usr/bin/env python3
"""Compare merkc's polymorphism analysis with Mercury's on modules.

Each module is checked by `merkc -fc1 -fdebug-dump-polymorphism` and by
`mmc --errorcheck-only --dump-hlds 50 --dump-hlds-options vx`, which
writes each procedure after mmc's polymorphism transformation: its hidden
type_info arguments in its head, the goals that make the type_infos its
calls pass, and which type each describes. Each procedure declared in the
module, by predicate or function and mode number, must have the same
hidden arguments in both, by their type variables' names, and its calls
that pass type_infos must pass them for the same types, in the order of
its body. The type_infos' variables, and where in the body mmc makes
them, are not compared; nor are the procedures merkc does not analyse
yet and instance methods.

With --infer, mmc infers the types and modes of predicates declared
without them (--infer-all), and merkc their modes (-finfer-modes).

A module with errors that stop mmc before its dump, such as type errors,
is reported, unless --allow-errors is given, and then skipped. The script
exits with status 1 if any procedure differs.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def canonical(procedure):
    """A procedure's hidden arguments and calls with its type variables
    named by their place among its hidden arguments, P1, P2, ..., since
    mmc and merkc name those of an inferred signature differently."""
    hidden, calls = procedure
    names = {name: "P%d" % (i + 1) for i, name in enumerate(hidden)}

    def rename(text):
        return re.sub(r"\b[A-Z]\w*\b", lambda m: names.get(m.group(0), m.group(0)), text)

    return ([names[n] for n in hidden],
            [(callee, [rename(t) for t in types]) for callee, types in calls])


def split_arguments(text):
    """The arguments of a call, split at its top-level commas."""
    out, depth, cur = [], 0, ""
    for c in text:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


TYPE_INFO_VAR = re.compile(r"^(TypeInfo|TypeCtorInfo)_\w+$")
CALL = re.compile(r"^\s*(?:\S+ = )?([a-z][\w.]*(?:\.'[^']*')?)\((.*)\)\.?$")
# The goals of mmc's that make type_infos, which are not calls.
MAKERS = ("type_info_cell_constructor", "typeclass_info_cell_constructor",
          "type_ctor_info", "type_info_const", "typeclass_info_const",
          "base_typeclass_info")


def mmc_procedures(dump):
    """Each procedure's hidden arguments and calls in mmc's dump, by (kind,
    name/arity, mode number), as (names, [(callee, types)])."""
    procedures = {}
    lines = dump.split("\n")
    pred = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"  % pred id \d+: (predicate|function) `(.*)'/(\d+)$", line)
        if m:
            kind = "pred" if m.group(1) == "predicate" else "func"
            pred = (kind, m.group(2), int(m.group(3)))
            i += 1
            continue
        m = re.match(r"  % mode number (\d+) ", line)
        if not (pred and m):
            i += 1
            continue
        mode = int(m.group(1))
        # The procedure's lines, up to the next.
        j = i + 1
        while j < len(lines) and not lines[j].startswith("  % pred id ") \
                and not re.match(r"  % mode number ", lines[j]):
            j += 1
        procedures[(pred[0], "%s/%d" % (pred[1], pred[2]), mode)] = \
            canonical(mmc_procedure(lines[i + 1:j]))
        i = j
    return procedures


def mmc_procedure(lines):
    described = {}
    for line in lines:
        # % T_1(number 1) -> type_info(TypeInfo_for_T_6) (number 6)
        m = re.match(r"% (\w+)\(number \d+\) -> type_info\((\w+)\)", line)
        if m:
            described[m.group(2)] = m.group(1)
        # % TypeInfo_31_31 (number 31) -> type_info for list.list(int)
        m = re.match(r"% (\w+) \(number \d+\) -> type_info for (.*)$", line)
        if m:
            described[m.group(1)] = m.group(2).strip()
        # TypeCtorInfo_27_27 = type_ctor_info("builtin", "int", 0)
        m = re.match(r'\s*(\w+) = type_ctor_info\("([^"]*)", "([^"]*)", 0\)$', line)
        if m:
            module, name = m.group(2), m.group(3)
            described[m.group(1)] = name if module == "builtin" else module + "." + name
    hidden = []
    calls = []
    head = True
    for line in lines:
        if line.startswith(":- mode "):
            head = True
            continue
        if head and not line.startswith("%") and line.strip():
            head = False
            m = re.match(r"^[\w.']+\((.*?)\)(?: = \w+)?(?: :-)?\.?$", line)
            if m:
                for argument in split_arguments(m.group(1)):
                    if argument not in described:
                        break
                    hidden.append(described[argument])
            continue
        m = CALL.match(line)
        if not m or line.lstrip().startswith("%") or m.group(1) in MAKERS:
            continue
        arguments = split_arguments(m.group(2))
        types = []
        for argument in arguments:
            if not TYPE_INFO_VAR.match(argument):
                break
            types.append(described.get(argument, "?"))
        if types:
            calls.append((m.group(1), types))
    return hidden, calls


def merkc_procedures(dump):
    procedures = {}
    key = None
    for line in dump.split("\n"):
        m = re.match(r"^\S.* (pred|func) (\S+)/(\d+) (?:inferred )?mode (\d+): ", line)
        if m:
            key = (m.group(1), "%s/%s" % (m.group(2), m.group(3)), int(m.group(4)))
            procedures[key] = ([], [])
            continue
        if line.startswith(("method ", "(default)")) and " mode " not in line:
            key = None
            continue
        if not key:
            continue
        m = re.match(r"^  not analysed: ", line)
        if m:
            del procedures[key]
            key = None
            continue
        m = re.match(r"^  type_info arguments: (.*)$", line)
        if m:
            procedures[key][0].extend(m.group(1).split(", "))
            continue
        m = re.match(r"^    \S+ (call|application) (?:pred|func) (\S+)/\d+: (.*)$", line)
        if m:
            types = [re.sub(r" \((argument|constant|cell)\)$", "", t)
                     for t in split_arguments(m.group(3))]
            procedures[key][1].append((m.group(2), types))
    return procedures


def show(procedure):
    hidden, calls = procedure
    out = ["    hidden: %s" % ", ".join(hidden)]
    for callee, types in calls:
        out.append("    %s: %s" % (callee, ", ".join(types)))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--infer", action="store_true",
                        help="infer types and modes of undeclared predicates")
    parser.add_argument("--verbose", action="store_true",
                        help="say how many procedures were compared")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    failed = False
    compared = 0
    for path in args.files:
        with open(path) as f:
            module = re.search(r"^:- module ([\w.]+)\.", f.read(), re.M).group(1)
        # mmc wants the file named after its module.
        copy = module + ".m"
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(path, os.path.join(tmp, copy))
            mmc = subprocess.run([args.mmc, "--errorcheck-only", "--dump-hlds", "50",
                                  "--dump-hlds-options", "vx", copy] +
                                 (["--infer-all"] if args.infer else []),
                                 cwd=tmp, capture_output=True, text=True)
            dump = os.path.join(tmp, module + ".hlds_dump.050-determinism")
            if not os.path.exists(dump):
                if not args.allow_errors:
                    print("%s: mmc stops before determinism analysis:\n%s" %
                          (path, mmc.stdout + mmc.stderr))
                    failed = True
                continue
            with open(dump) as f:
                theirs = mmc_procedures(f.read())
        merkc = subprocess.run([args.merkc, "-fc1", "-fdebug-dump-polymorphism", path] +
                               (["-finfer-modes"] if args.infer else []),
                               capture_output=True, text=True)
        ours = merkc_procedures(merkc.stdout)
        for key, procedure in ours.items():
            if key not in theirs:
                continue
            procedure = canonical(procedure)
            compared += 1
            if procedure != theirs[key]:
                print("%s: %s differs\n  mmc:\n%s\n  merkc:\n%s" %
                      (path, key, show(theirs[key]), show(procedure)))
                failed = True
    if args.verbose:
        print("%d procedures compared" % compared)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
