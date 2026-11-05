#!/usr/bin/env python3
"""Generate a program that io.writes every operator of the parser's table.

Each operator that a program can build is a constructor at each arity, written
alone, nested in itself on each side, and as an argument, so that compare-runs.py
can check the table in runtime/write.cpp against mmc's output. The operators that
are syntax in a data-term whichever way they are written cannot be built, so
nothing can write them.

Usage: generate-operator-program.py OPERATORS_DEF OUTPUT.m
"""

import re
import sys

UNBUILDABLE = {",", ";", "|", ".", "=>", "some", "-->", ":", ":-", "is", "^", "@"}


def quote(name):
    return "'" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"


def main():
    names = set()
    for line in open(sys.argv[1]):
        match = re.match(r'MERK_OPERATOR\("(.*)", \w+, \d+\)', line)
        if match:
            names.add(match[1].replace("\\\\", "\\"))
    constructors, goals = [], []
    for name in sorted(names - UNBUILDABLE):
        q = quote(name)
        constructors += ["(%s)" % q, q + "(o)", q + "(o, o)", q + "(o, o, o)"]
        goals += ["(%s)" % q, "%s(x)" % q, "%s(%s(x))" % (q, q), "%s(x, y)" % q,
                  "%s(%s(x, y), z)" % (q, q), "%s(x, %s(y, z))" % (q, q),
                  "%s(%s(x), z)" % (q, q), "%s(x, %s(z))" % (q, q),
                  "%s(x, y, z)" % q, "f(%s(x, y))" % q, "f(%s(x))" % q,
                  "%s(f(x))" % q]
    with open(sys.argv[2], "w") as out:
        out.write(""":- module operators_written.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type o ---> x ; y ; z ; f(o)
  ; %s.
:- pred w(o::in, io::di, io::uo) is det.
w(X, !IO) :- io.write(X, !IO), io.nl(!IO).
main(!IO) :-
    %s.
""" % ("\n  ; ".join(constructors),
            ",\n    ".join("w(%s, !IO)" % g for g in goals)))


if __name__ == "__main__":
    main()
