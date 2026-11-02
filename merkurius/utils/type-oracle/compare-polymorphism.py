#!/usr/bin/env python3
"""Compare merkc's polymorphism analysis with Mercury's on modules.

Each module is checked by `merkc -fc1 -fdebug-dump-polymorphism` and by
`mmc --errorcheck-only --dump-hlds 50 --dump-hlds-options vx`, which
writes each procedure after mmc's polymorphism transformation: its hidden
type_info and typeclass_info arguments in its head, where the type_infos
of its constrained type variables are, the goals that make the type_infos
and typeclass_infos its calls pass, and what each describes. Each
procedure declared in the module, by predicate or function and mode
number, and each instance method, by its instance, must have the same
hidden arguments in both, the same type_infos in typeclass_infos, and its
calls that pass them must pass the same, in the order of its body, each
described by its type or constraint and where it comes from: a hidden
argument, a constant, a cell, a place in a typeclass_info, a superclass of
another, or an instance's cell with the typeclass_infos of its
constraints. The variables, and where in the body mmc makes them, are not
compared; nor are the procedures merkc does not analyse yet.

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
    """A procedure with its type variables named by their places, P1, P2,
    ..., among its hidden type_info arguments and then its type_infos in
    typeclass_infos, since mmc and merkc name those of an inferred
    signature or an instance method differently."""
    hidden, found, calls = procedure
    order = [h for h in hidden if re.match(r"^[A-Z]\w*$", h)]
    order += [f.split(" ", 1)[0] for f in found]
    names = {}
    for name in order:
        names.setdefault(name, "P%d" % (len(names) + 1))

    def rename(text):
        return re.sub(r"\b[A-Z]\w*\b", lambda m: names.get(m.group(0), m.group(0)), text)

    return ([rename(h) for h in hidden], [rename(f) for f in found],
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


HIDDEN_VAR = re.compile(r"^(TypeInfo|TypeCtorInfo|TypeClassInfo)_\w+$")
CALL = re.compile(r"^\s*(?:\S+ = )?([a-z][\w.]*(?:\.'[^']*')?)\((.*)\)\.?$")
# The goals of mmc's that make type_infos and typeclass_infos, which are
# not calls.
MAKERS = ("type_info_cell_constructor", "typeclass_info_cell_constructor",
          "type_ctor_info", "type_info_const", "typeclass_info_const",
          "base_typeclass_info", "private_builtin.type_info_from_typeclass_info",
          "private_builtin.superclass_from_typeclass_info")


def instance_key(text):
    """An instance's text with its type variables' names left out."""
    return re.sub(r"\b[A-Z]\w*\b", "_", text)


def mmc_procedures(dump):
    """Each procedure's hidden arguments, type_infos in typeclass_infos and
    calls in mmc's dump, by (kind, name/arity, mode number), or for an
    instance method ("method", instance, name/arity, mode number)."""
    procedures = {}
    lines = dump.split("\n")
    pred = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"  % pred id \d+: (predicate|function) `(.*)'/(\d+)$", line)
        if m:
            kind = "pred" if m.group(1) == "predicate" else "func"
            pred = (kind, "%s/%s" % (m.group(2), m.group(3)))
            i += 1
            continue
        m = re.match(r"  % pred id \d+: instance method (predicate|function) "
                     r"`(.*)'/(\d+) for `(.*)'$", line)
        if m:
            pred = ("method", instance_key(m.group(4)), "%s/%s" % (m.group(2), m.group(3)))
            i += 1
            continue
        if re.match(r"  % pred id ", line):
            pred = None
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
        procedures[pred + (mode,)] = canonical(mmc_procedure(lines[i + 1:j]))
        i = j
    return procedures


def mmc_procedure(lines):
    # What each variable describes: its type or constraint.
    described = {}
    # How each is made: (maker, arguments).
    made = {}
    constants = {}
    # The type_info varmap's entries in typeclass_infos.
    found = []
    for line in lines:
        # % T_1(number 1) -> typeclass_info(TypeClassInfo_for_sh_4, 1) (number 4)
        m = re.match(r"% (\w+)\(number \d+\) -> typeclass_info\((\w+), (\d+)\)", line)
        if m:
            found.append((m.group(1), m.group(2), int(m.group(3))))
        m = re.match(r"% (\w+)\(number \d+\) -> type_info\((\w+)\)", line)
        if m:
            described[m.group(2)] = m.group(1)
        # % TypeInfo_31_31 (number 31) -> type_info for list.list(int)
        m = re.match(r"% (\w+) \(number \d+\) -> (?:type_info|typeclass_info) for (.*)$", line)
        if m:
            described[m.group(1)] = m.group(2).strip()
        m = re.match(r'\s*(\w+) = type_ctor_info\("([^"]*)", "([^"]*)", 0\)$', line)
        if m:
            module, name = m.group(2), m.group(3)
            described[m.group(1)] = name if module == "builtin" else module + "." + name
        m = re.match(r"\s*(\w+) = (type_info_const|type_ctor_info|type_info_cell_constructor|"
                     r"typeclass_info_const|typeclass_info_cell_constructor|"
                     r"base_typeclass_info)\((.*)\)$", line)
        if m:
            made[m.group(1)] = (m.group(2), split_arguments(m.group(3)))
        m = re.match(r"\s*private_builtin\.(type_info_from_typeclass_info|"
                     r"superclass_from_typeclass_info)\((\w+), (\w+), (\w+)\)$", line)
        if m:
            made[m.group(4)] = (m.group(1), [m.group(2), m.group(3)])
        m = re.match(r"\s*(PolyConst\w+) = (\d+)$", line)
        if m:
            constants[m.group(1)] = int(m.group(2))
    hidden_vars = []
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
                    hidden_vars.append(argument)
            continue
        m = CALL.match(line)
        if not m or line.lstrip().startswith("%") or m.group(1) in MAKERS:
            continue
        arguments = split_arguments(m.group(2))
        passed = []
        for argument in arguments:
            if not HIDDEN_VAR.match(argument):
                break
            passed.append(argument)
        if passed:
            calls.append((m.group(1), passed))

    def describe(var):
        what = described.get(var, "?")
        if var in hidden_vars:
            return "%s (argument)" % what
        maker, arguments = made.get(var, (None, []))
        if maker in ("type_info_const", "type_ctor_info", "typeclass_info_const"):
            return "%s (constant)" % what
        if maker == "type_info_cell_constructor":
            return "%s (cell)" % what
        if maker == "type_info_from_typeclass_info":
            return "%s (type_info %d of %s)" % (
                what, constants.get(arguments[1], 0), described.get(arguments[0], "?"))
        if maker == "superclass_from_typeclass_info":
            return "%s (superclass %d of %s)" % (
                what, constants.get(arguments[1], 0), describe(arguments[0]))
        if maker == "typeclass_info_cell_constructor":
            premises = [describe(a) for a in arguments[1:] if a.startswith("TypeClassInfo_")]
            return "%s (instance%s)" % (what, " with " + ", ".join(premises) if premises else "")
        return "%s (?)" % what

    hidden = [described[v] for v in hidden_vars]
    order = {v: k for k, v in enumerate(hidden_vars)}
    found.sort(key=lambda f: (order.get(f[1], len(order)), f[2]))
    found_text = ["%s (type_info %d of %s)" % (var, n, described.get(d, "?"))
                  for var, d, n in found]
    return hidden, found_text, [(callee, [describe(v) for v in passed])
                                for callee, passed in calls]


def merkc_procedures(dump):
    procedures = {}
    key = None
    for line in dump.split("\n"):
        m = re.match(r"^\S.* (pred|func) (\S+)/(\d+) (?:inferred )?mode (\d+): ", line)
        if m:
            key = (m.group(1), "%s/%s" % (m.group(2), m.group(3)), int(m.group(4)))
            procedures[key] = ([], [], [])
            continue
        m = re.match(r"^method (.*) (?:pred|func) (\S+) mode (\d+)$", line)
        if m:
            key = ("method", instance_key(m.group(1)), m.group(2), int(m.group(3)))
            procedures[key] = ([], [], [])
            continue
        if line.startswith("(default)") and " mode " not in line:
            key = None
            continue
        if not key:
            continue
        m = re.match(r"^  not analysed: ", line)
        if m:
            del procedures[key]
            key = None
            continue
        m = re.match(r"^  hidden arguments: (.*)$", line)
        if m:
            procedures[key][0].extend(split_arguments(m.group(1)))
            continue
        m = re.match(r"^  type_infos in typeclass_infos: (.*)$", line)
        if m:
            procedures[key][1].extend(split_arguments(m.group(1)))
            continue
        m = re.match(r"^    \S+ (call|application) (?:pred|func) (\S+)/\d+: (.*)$", line)
        if m:
            procedures[key][2].append((m.group(2), split_arguments(m.group(3))))
    return procedures


def show(procedure):
    hidden, found, calls = procedure
    out = ["    hidden: %s" % ", ".join(hidden)]
    if found:
        out.append("    in typeclass_infos: %s" % ", ".join(found))
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
