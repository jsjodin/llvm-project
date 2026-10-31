#!/usr/bin/env python3
"""Compare merkc's switch detection with Mercury's on modules.

Each module is checked by `merkc -fc1 -fdebug-dump-determinism` and by
`mmc --errorcheck-only --dump-hlds 45`, which writes mmc's form of each
procedure after switch detection and common subexpression elimination.
Each procedure declared in the module, by predicate or function and mode
number, must have the same structure in both: its switches, with their
variables, whether they can fail and their cases, in order, and its
disjunctions, if-then-elses and negations, nested as they are. The other
goals, conjunctions and `some` scopes are left out, and the variables
that mmc and merkc make, V_N, have no number. Inferred modes and instance
methods are not compared.

A module with errors that stop mmc before its dump, such as mode errors,
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


def normalise(label):
    return re.sub(r"\bV_\d+\b", "V", label)


def mmc_procedures(dump):
    """The structure of each procedure in mmc's dump, by (kind, name/arity,
    mode number), as a list of (depth, label)."""
    lines = dump.split("\n")
    procedures = {}
    pred = None
    mode = None
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
        if m:
            mode = int(m.group(1))
            i += 1
            continue
        if (pred and mode is not None and line.startswith(":- mode ")
                and i + 1 < len(lines) and re.match(r"^\S", lines[i + 1])):
            kind, name, arity = pred
            if kind == "func":
                arity -= 1
            key = "%s %s/%d mode %d" % (kind, name, arity, mode)
            body, i = mmc_body(lines, i + 2)
            procedures[key] = body
            mode = None
            continue
        i += 1
    return procedures


def mmc_body(lines, i):
    """Parses a body by its parentheses: each compound goal opens with `(`
    and closes with `)`, and its parts are separated by `;`, or by `then`
    and `else`."""
    root = {"kind": "conj", "label": None, "parts": [[]]}
    stack = [root]
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("  % pred id"):
        st = lines[i].strip()
        top = stack[-1]
        m = re.search(r"% (can_fail|cannot_fail) switch on `(.*)'", st)
        if st.startswith("("):
            node = {"kind": "conj", "label": None, "parts": [[]], "cases": []}
            if m:
                node["kind"] = "switch"
                node["label"] = "switch %s %s" % (m.group(2), m.group(1).replace("_", " "))
            elif "% disjunction" in st:
                node["kind"] = "disjunction"
            elif st.startswith("( if"):
                node["kind"] = "if-then-else"
            top["parts"][-1].append(node)
            stack.append(node)
        elif (re.match(r"^(not|\\\+|some \[.*\]|promise_\w+|\w+) ?\($", st) or st.endswith(" (")
              or re.search(r"= \((pred|func)\(.*:-$", st)):
            # A negation, a scope, or a lambda expression, whose body is
            # part of the structure.
            node = {"kind": "not" if st.startswith(("not", "\\+")) else "conj",
                    "label": None, "parts": [[]], "cases": []}
            top["parts"][-1].append(node)
            stack.append(node)
        elif st.startswith(")"):
            if len(stack) > 1:
                stack.pop()
        elif st == ";" or st in ("then", "else"):
            top["parts"].append([])
        elif "has functor" in st and top["kind"] == "switch":
            top["cases"].append("case " + st.split("has functor ")[1])
        i += 1
    out = []

    def emit(node, depth):
        kind = node["kind"]
        if kind == "switch":
            out.append((depth, normalise(node["label"])))
            for n, part in enumerate(node["parts"]):
                if n:
                    out.append((depth, "or"))
                if n < len(node["cases"]):
                    out.append((depth + 1, normalise(node["cases"][n])))
                for c in part:
                    emit(c, depth + 1)
        elif kind == "disjunction":
            out.append((depth, "disjunction"))
            for n, part in enumerate(node["parts"]):
                if n:
                    out.append((depth, "or"))
                for c in part:
                    emit(c, depth + 1)
        elif kind == "if-then-else":
            out.append((depth, "if-then-else"))
            for n, part in enumerate(node["parts"]):
                if n:
                    out.append((depth, ["then", "else"][n - 1] if n <= 2 else "?"))
                for c in part:
                    emit(c, depth + 1)
        elif kind == "not":
            out.append((depth, "not"))
            for part in node["parts"]:
                for c in part:
                    emit(c, depth + 1)
        else:
            for part in node["parts"]:
                for c in part:
                    emit(c, depth)

    emit(root, 0)
    return out, i


def merkc_procedures(dump):
    procedures = {}
    key = None
    nodes = []
    for line in dump.split("\n"):
        m = re.match(r"^\S.* (pred|func) (\S+)/(\d+) mode (\d+): ", line)
        if m:
            if key:
                procedures[key] = merkc_body(nodes)
            key = "%s %s/%s mode %s" % m.groups()
            nodes = []
            continue
        if line.startswith(("method ", "(default)")) or re.match(r"^\S.* inferred mode ", line):
            if key:
                procedures[key] = merkc_body(nodes)
            key = None
            nodes = []
            continue
        if key and line.strip():
            nodes.append(line)
    if key:
        procedures[key] = merkc_body(nodes)
    return procedures


def merkc_body(lines):
    # A tree by indentation: (text, children).
    root = ("", [])
    stack = [(-1, root)]
    for line in lines:
        ind = len(line) - len(line.lstrip())
        node = (line.strip(), [])
        while stack[-1][0] >= ind:
            stack.pop()
        stack[-1][1][1].append(node)
        stack.append((ind, node))
    out = []

    def emit(node, depth):
        text, children = node
        what = re.sub(r"^\d+:\d+ ", "", text)
        m = re.match(r"switch on (.*), (can fail|cannot fail)$", what)
        if m:
            out.append((depth, normalise("switch %s %s" % m.groups())))
            for n, case in enumerate(children):
                if n:
                    out.append((depth, "or"))
                out.append((depth + 1, normalise(case[0])))
                for c in case[1]:
                    emit(c, depth + 1)
            return
        if what == "disjunction":
            out.append((depth, "disjunction"))
            for n, branch in enumerate(children):
                if n:
                    out.append((depth, "or"))
                for c in branch[1]:
                    emit(c, depth + 1)
            return
        if what == "if-then-else":
            out.append((depth, "if-then-else"))
            for n, part in enumerate(children):
                if n:
                    out.append((depth, part[0]))
                for c in part[1]:
                    emit(c, depth + 1)
            return
        if what == "not":
            out.append((depth, "not"))
            for c in children:
                emit(c, depth + 1)
            return
        for c in children:
            emit(c, depth)

    for c in root[1]:
        emit(c, 0)
    return out


def show(labels):
    return "\n".join("    " + "  " * d + l for d, l in labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.files:
        name = os.path.basename(path)
        module = os.path.splitext(name)[0]
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(path, os.path.join(tmp, name))
            mmc = subprocess.run([args.mmc, "--errorcheck-only", "--dump-hlds", "45", name],
                                 cwd=tmp, capture_output=True, text=True)
            dump = os.path.join(tmp, module + ".hlds_dump.045-cse")
            if not os.path.exists(dump):
                if not args.allow_errors:
                    print("%s: mmc stops before switch detection:\n%s" %
                          (path, mmc.stdout + mmc.stderr))
                    failed = True
                continue
            with open(dump) as f:
                theirs = mmc_procedures(f.read())
        merkc = subprocess.run([args.merkc, "-fc1", "-fdebug-dump-determinism", path],
                               capture_output=True, text=True)
        if "error:" in merkc.stderr:
            print("%s: merkc reports errors that do not stop mmc:\n%s" % (path, merkc.stderr))
            failed = True
            continue
        ours = merkc_procedures(merkc.stdout)
        for key, labels in ours.items():
            if key not in theirs:
                continue
            if labels != theirs[key]:
                print("%s: %s differs\n  mmc:\n%s\n  merkc:\n%s" %
                      (path, key, show(theirs[key]), show(labels)))
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
