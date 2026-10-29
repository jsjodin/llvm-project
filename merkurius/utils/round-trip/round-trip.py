#!/usr/bin/env python3
"""Run the unparser's round-trip tests on Mercury files that convert
without errors (docs/MercuryMLIRCompilerandRuntimeDesign.md, "Unparser and
round-trip tests").

2. Exact round trip: `merkc -fc1 -fdebug-unparse-exact` gives back the
   file, byte for byte, and reports nothing: no node of the parse tree
   misplaces a token, and every variable and literal is held by a node.
4. Meaning preserved: the file and its unparsed form C1 have the same parse
   tree, as printed by -fdebug-dump-parse-tree-no-sema, which shows no
   locations or trivia. Until the unparser has its canonical layout, C1 is
   the minimal layout: the file without comments or line number directives,
   with each token's layout reduced to one space or newline.

The script exits with status 1 and reports every file that fails a test.
"""

import argparse
import difflib
import subprocess
import sys


def merkc(merkc_path, action, path=None, text=None):
    """Runs merkc -fc1 with `action` on a file, or on `text` from stdin."""
    args = [merkc_path, "-fc1", action, path if path else "-"]
    return subprocess.run(args, input=text, capture_output=True)


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
    c1 = merkc(merkc_path, "-fdebug-unparse-exact-minimal-layout", path)
    if c1.returncode != 0:
        problems.append(
            "the minimal layout failed:\n" + c1.stderr.decode(errors="replace")
        )
        return problems
    again = merkc(merkc_path, "-fdebug-dump-parse-tree-no-sema", text=c1.stdout)
    if again.returncode != 0 or again.stderr:
        problems.append(
            "the minimal layout does not convert:\n"
            + again.stderr.decode(errors="replace")
        )
    elif again.stdout != tree.stdout:
        diff = difflib.unified_diff(
            tree.stdout.decode(errors="replace").splitlines(),
            again.stdout.decode(errors="replace").splitlines(),
            "file",
            "minimal layout",
            lineterm="",
            n=2,
        )
        problems.append(
            "the minimal layout has a different parse tree:\n"
            + "\n".join(list(diff)[:40])
        )
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merkc", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.files:
        for problem in check(args.merkc, path):
            print("%s: %s" % (path, problem))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
