#!/usr/bin/env python3
"""Run the unparser's round-trip tests on Mercury files that convert
without errors (docs/MercuryMLIRCompilerandRuntimeDesign.md, "Unparser and
round-trip tests").

2. Exact round trip: `merkc -fc1 -fdebug-unparse-exact` gives back the
   file, byte for byte, and reports nothing: no node of the parse tree
   misplaces a token, and every variable and literal is held by a node.
3. Canonical fixed point: the canonical form C1 of the file
   (-fdebug-unparse-no-sema) is its own canonical form.
4. Meaning preserved: the file, C1, and the file with every operator term
   parenthesized (-fdebug-unparse-parenthesized-no-sema) have the same parse
   tree, as printed by -fdebug-dump-parse-tree-no-sema, which shows no
   locations or trivia.

With --openmp, %$omp directives are read as directives, not comments.

The script exits with status 1 and reports every file that fails a test.
"""

import argparse
import difflib
import subprocess
import sys


OPTIONS = []


def merkc(merkc_path, action, path=None, text=None):
    """Runs merkc -fc1 with `action` on a file, or on `text` from stdin."""
    args = [merkc_path, "-fc1"] + OPTIONS + [action, path if path else "-"]
    return subprocess.run(args, input=text, capture_output=True)


def diff(a, b, a_name, b_name):
    """The start of a unified diff of two outputs."""
    lines = difflib.unified_diff(
        a.decode(errors="replace").splitlines(),
        b.decode(errors="replace").splitlines(),
        a_name,
        b_name,
        lineterm="",
        n=2,
    )
    return "\n".join(list(lines)[:40])


def check(merkc_path, path):
    """Returns the problems found in one file, as strings."""
    problems = []
    with open(path, "rb") as f:
        original = f.read()

    exact = merkc(merkc_path, "-fdebug-unparse-exact", path)
    if exact.returncode != 0 or exact.stderr:
        problems.append(
            "the exact unparser reported:\n" + exact.stderr.decode(errors="replace")
        )
    elif exact.stdout != original:
        problems.append("the exact unparser did not give back the file")

    tree = merkc(merkc_path, "-fdebug-dump-parse-tree-no-sema", path)
    c1 = merkc(merkc_path, "-fdebug-unparse-no-sema", path)
    if c1.returncode != 0:
        problems.append(
            "the canonical unparser failed:\n" + c1.stderr.decode(errors="replace")
        )
        return problems
    c2 = merkc(merkc_path, "-fdebug-unparse-no-sema", text=c1.stdout)
    if c2.stdout != c1.stdout:
        problems.append(
            "the canonical form is not a fixed point:\n"
            + diff(c1.stdout, c2.stdout, "canonical", "canonical twice")
        )
    parenthesized = merkc(merkc_path, "-fdebug-unparse-parenthesized-no-sema", path)
    for name, text in (("canonical form", c1.stdout),
                       ("parenthesized form", parenthesized.stdout)):
        again = merkc(merkc_path, "-fdebug-dump-parse-tree-no-sema", text=text)
        if again.returncode != 0 or again.stderr:
            problems.append(
                "the %s does not convert:\n%s"
                % (name, again.stderr.decode(errors="replace"))
            )
        elif again.stdout != tree.stdout:
            problems.append(
                "the %s has a different parse tree:\n%s"
                % (name, diff(tree.stdout, again.stdout, "file", name))
            )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--openmp", action="store_true")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.openmp:
        OPTIONS.append("-fopenmp")
    failed = False
    for path in args.files:
        for problem in check(args.merkc, path):
            print("%s: %s" % (path, problem))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
