#!/usr/bin/env python3
"""Write Mercury items that combine every pair of builtin operators.

The items go to standard output, one per line, for compare-terms.py to read
with both merkc and Mercury's term parser: every infix operator with every
infix operator, prefix operators before and after infix ones, prefix
operators applied to each other, the binary prefix operators, operators as
constants, and backquoted operators among the others. Whether each item is
valid does not matter; the two readers must agree on it.
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "..", "..", "lib", "Parser", "operators.def")


def operators():
    """(name, specifier, priority) for each row of the table."""
    text = open(TABLE).read()
    rows = re.findall(r'^MERK_OPERATOR\("((?:[^"\\]|\\.)*)", (\w+), (\d+)\)', text, re.M)
    return [(name.encode().decode("unicode_escape"), spec, int(p)) for name, spec, p in rows]


def spelled(name):
    # '.' followed by a space would be an end token.
    return name


def infix(left, op, right):
    if op == ".":
        return f"{left}.{right}"
    return f"{left} {op} {right}"


def main():
    rows = operators()
    infixes = sorted({n for n, s, _ in rows if s in ("XFX", "XFY", "YFX")})
    prefixes = sorted({n for n, s, _ in rows if s in ("FX", "FY")})
    binary_prefixes = sorted({n for n, s, _ in rows if s == "FXY"})
    everything = sorted({n for n, _, _ in rows})
    items = []
    for a in infixes:
        for b in infixes:
            items.append(infix(infix("a", a, "b"), b, "c"))
    for p in prefixes:
        for b in infixes:
            items.append(infix(f"{p} a", b, "b"))
            items.append(infix("a", b, f"{p} b"))
    for p in prefixes:
        for q in prefixes:
            items.append(f"{p} {q} a")
    for f in binary_prefixes:
        items.append(f"{f} [x] a")
        for b in infixes:
            items.append(infix(f"{f} [x] a", b, "b"))
            items.append(infix("a", b, f"{f} [x] b"))
    for op in everything:
        items.extend([f"f({op})", f"[{op}]", f"({op})", f"x = {op}", f"{op}",
                      f"f({op}, a)", f"f(a, {op})", f"x = ({op})"])
    for b in infixes:
        items.append(infix("a `f` b", b, "c"))
        items.append(infix("a", b, "b `f` c"))
    for item in items:
        # A space, so that an item ending in a graphic name still ends.
        sys.stdout.write(item + " .\n")


if __name__ == "__main__":
    main()
