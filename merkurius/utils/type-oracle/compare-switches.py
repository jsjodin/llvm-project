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

With --determinism, the dump is mmc's after determinism analysis
(`--dump-hlds 50 --dump-hlds-options d`), and each switch, disjunction,
if-then-else and negation has its determinism too, as has each procedure,
which is the one inferred for it; and the lines of mmc's error messages,
but not its warnings, must be those of merkc's errors and their notes. A
procedure that merkc does not infer a determinism for is not compared.

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


determinism = False


def normalise(label):
    return re.sub(r"\bV_\d+\b", "V", label)


def mmc_procedures(dump):
    """The structure of each procedure in mmc's dump, by (kind, name/arity,
    mode number), as a list of (depth, label); with --determinism, the
    first label is the procedure's inferred determinism."""
    lines = dump.split("\n")
    procedures = {}
    pred = None
    mode = None
    detism = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"  % mode number (\d+) \((\w+)\)", line)
        if m:
            detism = m.group(2)
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
            if determinism:
                body = [(0, "inferred " + detism)] + body
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
    # The determinism of the goal after it.
    detism = None
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("  % pred id"):
        st = lines[i].strip()
        top = stack[-1]
        m = re.match(r"% determinism: (\w+)$", st)
        if m:
            detism = m.group(1)
            i += 1
            continue
        # mmc marks a goal in a context where only its first solution is
        # wanted, as in a negation, with a committed choice determinism,
        # which says nothing more about the goal.
        tag = (" [%s]" % re.sub(r"^cc_", "", detism)) if determinism and detism else ""
        m = re.search(r"% (can_fail|cannot_fail) switch on `(.*)'", st)
        if st.startswith("("):
            node = {"kind": "conj", "label": None, "parts": [[]], "cases": [], "tag": tag}
            if m:
                node["kind"] = "switch"
                node["label"] = "switch %s %s" % (m.group(2), m.group(1).replace("_", " ")) + tag
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
                    "label": None, "parts": [[]], "cases": [], "tag": tag}
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
            out.append((depth, "disjunction" + node.get("tag", "")))
            for n, part in enumerate(node["parts"]):
                if n:
                    out.append((depth, "or"))
                for c in part:
                    emit(c, depth + 1)
        elif kind == "if-then-else":
            out.append((depth, "if-then-else" + node.get("tag", "")))
            for n, part in enumerate(node["parts"]):
                if n:
                    out.append((depth, ["then", "else"][n - 1] if n <= 2 else "?"))
                for c in part:
                    emit(c, depth + 1)
        elif kind == "not":
            out.append((depth, "not" + node.get("tag", "")))
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
                procedures[key] = merkc_body(nodes, inferred)
            key = "%s %s/%s mode %s" % m.groups()
            nodes = []
            inferred = re.search(r", inferred (\w+)$", line)
            if determinism and not inferred:
                # Not inferred yet, so not compared.
                key = None
            continue
        if line.startswith(("method ", "(default)")) or re.match(r"^\S.* inferred mode ", line):
            if key:
                procedures[key] = merkc_body(nodes, inferred)
            key = None
            nodes = []
            continue
        if key and line.strip():
            nodes.append(line)
    if key:
        procedures[key] = merkc_body(nodes, inferred)
    return procedures


def merkc_body(lines, inferred=None):
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
        m = re.search(r" \[(\w+)\]$", what)
        tag = " [%s]" % m.group(1) if m and determinism else ""
        what = re.sub(r" \[\w+\]$", "", what)
        m = re.match(r"switch on (.*), (can fail|cannot fail)$", what)
        if m:
            out.append((depth, normalise("switch %s %s" % m.groups()) + tag))
            for n, case in enumerate(children):
                if n:
                    out.append((depth, "or"))
                out.append((depth + 1, normalise(case[0])))
                for c in case[1]:
                    emit(c, depth + 1)
            return
        if what == "disjunction":
            out.append((depth, "disjunction" + tag))
            for n, branch in enumerate(children):
                if n:
                    out.append((depth, "or"))
                for c in branch[1]:
                    emit(c, depth + 1)
            return
        if what == "if-then-else":
            out.append((depth, "if-then-else" + tag))
            for n, part in enumerate(children):
                if n:
                    out.append((depth, part[0]))
                for c in part[1]:
                    emit(c, depth + 1)
            return
        if what == "not":
            out.append((depth, "not" + tag))
            for c in children:
                emit(c, depth + 1)
            return
        for c in children:
            emit(c, depth)

    if determinism and inferred:
        out.append((0, "inferred " + inferred.group(1)))
    for c in root[1]:
        emit(c, 0)
    return out


# merkc's determinism errors, which mmc reports after switch detection.
DETERMINISM_ERROR = re.compile(
    r"error: (the determinism declaration of|the lambda expression is declared|invalid determinism)")


def mmc_error_lines(output, name):
    """The lines of mmc's messages but its warnings: each message starts
    with a line whose text is not indented, and may go on at other lines."""
    messages = []
    prefix = re.compile(re.escape(name) + r":(\d+): (.*)$")
    for raw in output.splitlines():
        m = prefix.match(raw)
        if not m:
            continue
        line, text = int(m.group(1)), m.group(2)
        if text.startswith(" ") and messages:
            messages[-1].append((line, text.strip()))
        else:
            messages.append([(line, text.strip())])
    lines = set()
    for message in messages:
        texts = [t for _, t in message]
        # A warning's header, "In clause for ...:", may take several lines.
        kind = next((t for t in texts if re.match(r"(?i)(warning|error)\b", t)), "")
        if texts[0].startswith("Warning") or kind.lower().startswith("warning"):
            continue
        lines.update(l for l, _ in message)
    return lines


def merkc_error_lines(stderr, name):
    return {int(m.group(1)) for m in re.finditer(
        r"^(?:.*/)?" + re.escape(name) + r":(\d+):\d+: (?:error|because):", stderr, re.M)}


def show(labels):
    return "\n".join("    " + "  " * d + l for d, l in labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--determinism", action="store_true",
                        help="compare determinisms and determinism errors too")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    global determinism
    determinism = args.determinism
    stage = ["--dump-hlds", "50", "--dump-hlds-options", "d"] if determinism else ["--dump-hlds", "45"]
    suffix = ".hlds_dump.050-determinism" if determinism else ".hlds_dump.045-cse"
    failed = False
    for path in args.files:
        name = os.path.basename(path)
        with open(path) as f:
            module = re.search(r"^:- module ([\w.]+)\.", f.read(), re.M).group(1)
        # mmc wants the file named after its module.
        copy = module + ".m"
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(path, os.path.join(tmp, copy))
            mmc = subprocess.run([args.mmc, "--errorcheck-only"] + stage + [copy],
                                 cwd=tmp, capture_output=True, text=True)
            dump = os.path.join(tmp, module + suffix)
            if not os.path.exists(dump):
                if not args.allow_errors:
                    print("%s: mmc stops before switch detection:\n%s" %
                          (path, mmc.stdout + mmc.stderr))
                    failed = True
                continue
            else:
                with open(dump) as f:
                    theirs = mmc_procedures(f.read())
        merkc = subprocess.run([args.merkc, "-fc1", "-fdebug-dump-determinism", path],
                               capture_output=True, text=True)
        others = [l for l in merkc.stderr.splitlines()
                  if " error: " in l and not DETERMINISM_ERROR.search(l)]
        if others:
            print("%s: merkc reports errors that do not stop mmc:\n%s" % (path, merkc.stderr))
            failed = True
            continue
        if determinism:
            theirs_lines = mmc_error_lines(mmc.stdout + mmc.stderr, copy)
            ours_lines = merkc_error_lines(merkc.stderr, name)
            if ours_lines != theirs_lines:
                print("%s: error lines differ: mmc only %s, merkc only %s" %
                      (path, sorted(theirs_lines - ours_lines), sorted(ours_lines - theirs_lines)))
                failed = True
        ours = merkc_procedures(merkc.stdout)
        for key, labels in ours.items():
            if key not in theirs:
                continue
            if labels != theirs[key]:
                # With --determinism, whether the procedure's own
                # determinism differs, or only its goals'.
                which = ("determinism " if determinism and labels[:1] != theirs[key][:1]
                         else "")
                print("%s: %s %sdiffers\n  mmc:\n%s\n  merkc:\n%s" %
                      (path, key, which, show(theirs[key]), show(labels)))
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
