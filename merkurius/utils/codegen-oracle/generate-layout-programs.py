#!/usr/bin/env python3
"""Generate random programs for compare-runs.py that exercise terms' layouts.

Each module defines a few types at random (design document, "Type
representation"): types of constants only; types of one constructor of one
argument, which are their argument, chained through each other; and types of
constants and constructors with arguments, whose number is chosen around the
primary tags' boundaries, 7, 8 and 9 constructors with arguments, with and
without constants, so that some share the last tag and have secondary tags.
An argument is an int, a float, a char or a sized integer, which are
narrower than a word, an earlier type, the type itself, a list or a tuple.
Each value printed first goes through polymorphic code, where it is a word,
and must come out as it went in. For each type the module
prints every constructor (a switch), gives each one a code (a function by
clauses), and in main builds values both as ground terms, which are static
cells, and with some of their integers computed at run time, so that they
are allocated; it prints them, unifies each static value with its
allocated twin and with another value, and tests a value's constructor,
taking it apart. The modules are det and print only, so that their output
is the comparison.

Usage: generate-layout-programs.py [--seed N] [--count N] DIR
"""

import argparse
import os
import random

# The number of constructors with arguments: around the boundaries of the
# eight primary tags, of which constants take tag 0.
FUNCTOR_COUNTS = [1, 2, 3, 6, 7, 7, 8, 8, 9, 9, 10, 13]
CONSTANT_COUNTS = [0, 0, 1, 1, 2, 3]
MAX_DEPTH = 3
# The sized integers, with their literals' suffixes and ranges.
SIZED = {"int8": ("i8", -128, 127), "uint8": ("u8", 0, 255),
         "int16": ("i16", -32768, 32767), "uint32": ("u32", 0, 4000000000)}


class Type:
    def __init__(self, index, kind):
        self.index = index
        self.kind = kind  # "enum", "notag" or "cells"
        self.name = "t%d" % index
        # Each constructor is (name, [field types]); constants have none.
        self.constructors = []

    def base(self):
        """The constructors that do not contain the type itself directly."""
        return [c for c in self.constructors
                if not any(contains(f, self) for f in c[1])]


def contains(field, t):
    """Whether a field type is t, or a tuple with t in it. A list of t is
    not, since [] ends it."""
    if field == ("type", t):
        return True
    if field[0] == "tuple":
        return any(contains(f, t) for f in field[1:])
    return False


def type_string(field):
    kind = field[0]
    if kind in ("int", "float", "char") or kind in SIZED:
        return kind
    if kind == "type":
        return field[1].name
    if kind == "list":
        return "list(%s)" % type_string(field[1])
    return "{%s}" % ", ".join(type_string(f) for f in field[1:])


def key(field):
    """A name for a field type, for the helpers that show it."""
    kind = field[0]
    if kind in ("int", "float", "char") or kind in SIZED:
        return kind
    if kind == "type":
        return field[1].name
    if kind == "list":
        return "list_" + key(field[1])
    return "tup_" + "_".join(key(f) for f in field[1:])


# Polymorphic code each value goes through: a bare type variable, a list(T),
# a notag type, a cell with other fields, and the type_infos these build.
POLYMORPHIC_CODE = """\
:- type poly_box(T) ---> poly_box(T).
:- type poly_tree(K, V) ---> poly_leaf ; poly_node(poly_tree(K, V), K, V).

:- func poly_id(T) = T.
poly_id(X) = X.

:- func poly_second(A, B) = B.
poly_second(_, B) = B.

:- func poly_last(list(T), T) = T.
poly_last([], D) = D.
poly_last([H | T], _) = poly_last(T, H).

:- func poly_rev(list(T), list(T)) = list(T).
poly_rev([], A) = A.
poly_rev([H | T], A) = poly_rev(T, [H | A]).

:- func poly_unbox(poly_box(T)) = T.
poly_unbox(poly_box(X)) = X.

:- func poly_value(poly_tree(K, V), V) = V.
poly_value(poly_leaf, D) = D.
poly_value(poly_node(_, _, V), _) = V.

:- func poly_head(list(T)) = T.
poly_head(L) = poly_last(L, poly_head_of(L)).

:- func poly_head_of(list(T)) = T.
poly_head_of([H | _]) = H.
poly_head_of([]) = poly_head_of([]).

:- func poly_value_of(T) = T.
poly_value_of(X) = poly_value(poly_node(poly_leaf, "key", X), X).

:- pred poly_same(T::in, T::in) is semidet.
poly_same(X, X).

:- pred print_order(comparison_result::in, io::di, io::uo) is det.
print_order((=), !IO) :- io.write_string("=", !IO).
print_order((<), !IO) :- io.write_string("<", !IO).
print_order((>), !IO) :- io.write_string(">", !IO).

:- pred poly_order(T::in, T::in, io::di, io::uo) is det.
poly_order(X, Y, !IO) :-
    compare(R, X, Y),
    print_order(R, !IO).

:- func poly_nested(T) = T.
poly_nested(X) = poly_last(poly_last([[X]], [X]), poly_unbox(poly_box(X))).
"""
POLYMORPHIC_ROUTES = [
    "poly_id(%s)",
    "poly_second(1.5, %s)",
    "poly_head(poly_rev([%s], []))",
    "poly_value_of(%s)",
    "poly_unbox(poly_box(%s))",
    "poly_unbox(poly_id(poly_box(%s)))",
    "poly_nested(%s)",
]


class Generator:
    def __init__(self, rng):
        self.rng = rng
        self.types = []
        self.helpers = {}  # key -> (field, code)
        self.values = []  # the functions that make the values
        self.fresh = 0

    def var(self, prefix):
        self.fresh += 1
        return "%s%d" % (prefix, self.fresh)

    # Types.

    def field_type(self, t, allow_self, depth=0):
        rng = self.rng
        choices = ["int", "int", "float", "char", "sized"]
        if self.types:
            choices += ["earlier", "earlier"]
        if allow_self:
            choices += ["self"]
        if depth == 0:
            choices += ["list", "tuple"]
        choice = rng.choice(choices)
        if choice in ("int", "float", "char"):
            return (choice,)
        if choice == "sized":
            return (rng.choice(sorted(SIZED)),)
        if choice == "earlier":
            return ("type", rng.choice(self.types))
        if choice == "self":
            return ("type", t)
        if choice == "list":
            # A list of the type itself ends with [], so it may be one.
            return ("list", self.field_type(t, True, depth + 1))
        return ("tuple", ("int",), self.field_type(t, allow_self, depth + 1))

    def make_type(self, index):
        rng = self.rng
        kind = rng.choices(["enum", "notag", "cells"], [15, 15, 70])[0]
        t = Type(index, kind)
        if kind == "enum":
            for j in range(rng.randint(1, 9)):
                t.constructors.append(("k%d_%d" % (index, j), []))
        elif kind == "notag":
            t.constructors.append(
                ("w%d" % index, [self.field_type(t, False, 1)]))
        else:
            constants = rng.choice(CONSTANT_COUNTS)
            functors = rng.choice(FUNCTOR_COUNTS)
            names = (["k%d_%d" % (index, j) for j in range(constants)] +
                     ["c%d_%d" % (index, j) for j in range(functors)])
            rng.shuffle(names)
            for name in names:
                if name.startswith("k"):
                    t.constructors.append((name, []))
                else:
                    arity = rng.randint(1, 3)
                    t.constructors.append(
                        (name, [self.field_type(t, True) for _ in range(arity)]))
            # Some constructor must end a value of the type.
            if not t.base():
                name, fields = t.constructors[0]
                t.constructors[0] = (
                    name, [self.field_type(t, False) for _ in fields])
        self.types.append(t)
        return t

    def declaration(self, t):
        lines = [":- type %s" % t.name]
        for i, (name, fields) in enumerate(t.constructors):
            prefix = "    --->    " if i == 0 else "    ;       "
            if fields:
                lines.append("%s%s(%s)" % (
                    prefix, name, ", ".join(type_string(f) for f in fields)))
            else:
                lines.append(prefix + name)
        return "\n".join(lines) + ".\n"

    # Values: a value is (field, payload), which is rendered either static,
    # with every integer a literal, or with some integers computed from N,
    # which is 0 at run time, so that the term is allocated.

    def value(self, field, depth):
        rng = self.rng
        kind = field[0]
        if kind == "int":
            return (field, rng.randint(-50, 50))
        if kind == "float":
            return (field, rng.randint(-20, 20) + rng.choice([0.0, 0.25, 0.5, 0.75]))
        if kind == "char":
            return (field, rng.choice("abcdefghijklmnopqrstuvwxyz"))
        if kind in SIZED:
            return (field, rng.randint(*SIZED[kind][1:]))
        if kind == "list":
            length = 0 if depth >= MAX_DEPTH else rng.randint(0, 3)
            return (field, [self.value(field[1], depth + 1) for _ in range(length)])
        if kind == "tuple":
            return (field, [self.value(f, depth + 1) for f in field[1:]])
        t = field[1]
        constructors = t.base() if depth >= MAX_DEPTH else t.constructors
        name, fields = rng.choice(constructors)
        return (field, (name, [self.value(f, depth + 1) for f in fields]))

    def render(self, value, dynamic):
        field, payload = value
        kind = field[0]
        if kind == "int":
            if dynamic and self.rng.random() < 0.5:
                return "(N + %d)" % payload if payload >= 0 else "(N - %d)" % -payload
            return str(payload) if payload >= 0 else "(%d)" % payload
        if kind == "float":
            text = repr(float(payload))
            return text if payload >= 0 else "(%s)" % text
        if kind == "char":
            return "'%s'" % payload
        if kind in SIZED:
            if dynamic and self.rng.random() < 0.5:
                return "%s.cast_from_int(N + %d)" % (kind, payload)
            text = "%d%s" % (payload, SIZED[kind][0])
            return text if payload >= 0 else "(%s)" % text
        if kind == "list":
            return "[%s]" % ", ".join(self.render(v, dynamic) for v in payload)
        if kind == "tuple":
            return "{%s}" % ", ".join(self.render(v, dynamic) for v in payload)
        name, arguments = payload
        if not arguments:
            return name
        return "%s(%s)" % (name, ", ".join(self.render(v, dynamic) for v in arguments))

    # Printing.

    def show(self, field, variable):
        """A goal that prints a value of the field type."""
        kind = field[0]
        if kind == "int":
            return "io.write_int(%s, !IO)" % variable
        if kind == "float":
            return "io.write_float(%s, !IO)" % variable
        if kind == "char":
            return "io.write_char(%s, !IO)" % variable
        if kind in SIZED:
            return "io.write_%s(%s, !IO)" % (kind, variable)
        if kind == "type":
            return "show_%s(%s, !IO)" % (field[1].name, variable)
        self.helper(field)
        return "show_%s(%s, !IO)" % (key(field), variable)

    def helper(self, field):
        name = key(field)
        if name in self.helpers:
            return
        self.helpers[name] = None
        mtype = type_string(field)
        if field[0] == "list":
            element = self.show(field[1], "X")
            code = (
                ":- pred show_{n}({t}::in, io::di, io::uo) is det.\n"
                "show_{n}([], !IO) :-\n"
                "    io.write_string(\"[]\", !IO).\n"
                "show_{n}([X | Xs], !IO) :-\n"
                "    {e},\n"
                "    io.write_string(\" | \", !IO),\n"
                "    show_{n}(Xs, !IO).\n").format(n=name, t=mtype, e=element)
        else:
            parts = ["A%d" % i for i in range(len(field) - 1)]
            goals = ["io.write_string(\"{\", !IO)"]
            for i, (part, f) in enumerate(zip(parts, field[1:])):
                if i:
                    goals.append("io.write_string(\", \", !IO)")
                goals.append(self.show(f, part))
            goals.append("io.write_string(\"}\", !IO)")
            code = (
                ":- pred show_{n}({t}::in, io::di, io::uo) is det.\n"
                "show_{n}({{{p}}}, !IO) :-\n"
                "    {g}.\n").format(n=name, t=mtype, p=", ".join(parts),
                                     g=",\n    ".join(goals))
        self.helpers[name] = code

    def show_pred(self, t):
        lines = [":- pred show_%s(%s::in, io::di, io::uo) is det." % (t.name, t.name)]
        for name, fields in t.constructors:
            if not fields:
                lines.append("show_%s(%s, !IO) :-\n    io.write_string(\"%s\", !IO)."
                             % (t.name, name, name))
                continue
            arguments = ["A%d" % i for i in range(len(fields))]
            goals = ["io.write_string(\"%s(\", !IO)" % name]
            for i, (a, f) in enumerate(zip(arguments, fields)):
                if i:
                    goals.append("io.write_string(\", \", !IO)")
                goals.append(self.show(f, a))
            goals.append("io.write_string(\")\", !IO)")
            lines.append("show_%s(%s(%s), !IO) :-\n    %s." % (
                t.name, name, ", ".join(arguments), ",\n    ".join(goals)))
        return "\n".join(lines) + "\n"

    def code_func(self, t):
        lines = [":- func code_%s(%s) = int." % (t.name, t.name)]
        for i, (name, fields) in enumerate(t.constructors):
            pattern = name if not fields else "%s(%s)" % (
                name, ", ".join("_" for _ in fields))
            lines.append("code_%s(%s) = %d." % (t.name, pattern, i))
        return "\n".join(lines) + "\n"

    # Main.

    def routed(self, variable, goals):
        """A new variable with the value of variable, by way of polymorphic
        code, in which it is a word: it must come out as it went in."""
        route = self.rng.choice(POLYMORPHIC_ROUTES)
        result = self.var("P")
        goals.append("%s = %s" % (result, route % variable))
        return result

    def tests(self, t):
        rng = self.rng
        goals = []
        values = [self.value(("type", t), 0) for _ in range(rng.randint(2, 4))]
        names = []
        for value in values:
            s, d = self.var("S"), self.var("D")
            names.append((s, d))
            # Each value is a function's, so that mode analysis knows only
            # that it is ground, as a value read from input would be.
            static, dynamic = self.render(value, False), self.render(value, True)
            self.values.append(":- func %s = %s.\n%s = %s.\n" % (
                s.lower(), t.name, s.lower(), static))
            self.values.append(":- func %s(int) = %s.\n%s(%s) = %s.\n" % (
                d.lower(), t.name, d.lower(), "N" if "N" in dynamic else "_",
                dynamic))
            goals.append("%s = %s" % (s, s.lower()))
            goals.append("%s = %s(N)" % (d, d.lower()))
            routed = self.routed(d, goals)
            goals.append(self.show(("type", t), routed))
            goals.append("io.nl(!IO)")
            goals.append("io.write_int(code_%s(%s), !IO)" % (t.name, routed))
            goals.append("io.nl(!IO)")
            # Equal, though one is static and the other allocated.
            goals.append("( if %s = %s then io.write_string(\"eq\", !IO) "
                         "else io.write_string(\"ne\", !IO) )" % (s, d))
            goals.append("io.nl(!IO)")
            # And by the type_info's descriptor.
            goals.append("( if poly_same(%s, %s) then io.write_string(\"eq\", !IO) "
                         "else io.write_string(\"ne\", !IO) )" % (s, d))
            goals.append("io.nl(!IO)")
        for (s1, _), (_, d2) in zip(names, names[1:]):
            goals.append("( if %s = %s then io.write_string(\"eq\", !IO) "
                         "else io.write_string(\"ne\", !IO) )" % (s1, d2))
            goals.append("( if poly_same([%s], [%s]) then io.write_string(\"eq\", !IO) "
                         "else io.write_string(\"ne\", !IO) )" % (s1, d2))
            # Their order, at the known type and by the descriptor.
            goals.append("compare(%s_order, %s, %s)" % (s1, s1, d2))
            goals.append("print_order(%s_order, !IO)" % s1)
            goals.append("poly_order(%s, %s, !IO)" % (s1, d2))
            goals.append("poly_order([%s], [%s], !IO)" % (s1, d2))
            goals.append("io.nl(!IO)")
        # A test of a constructor that takes it apart.
        _, d = rng.choice(names)
        name, fields = rng.choice(t.constructors)
        arguments = [self.var("A") for _ in fields]
        pattern = name if not fields else "%s(%s)" % (name, ", ".join(arguments))
        then = ["io.write_string(\"is %s\", !IO)" % name]
        for a, f in zip(arguments, fields):
            then.append("io.write_char(' ', !IO)")
            then.append(self.show(f, self.routed(a, then)))
        goals.append("( if %s = %s then\n        %s\n    else\n        "
                     "io.write_string(\"not %s\", !IO)\n    )" % (
                         d, pattern, ",\n        ".join(then), name))
        goals.append("io.nl(!IO)")
        return goals

    def module(self, name):
        rng = self.rng
        for i in range(rng.randint(3, 6)):
            self.make_type(i + 1)
        parts = [
            ":- module %s.\n" % name,
            ":- interface.\n",
            ":- import_module io.\n",
            ":- pred main(io::di, io::uo) is det.\n",
            ":- implementation.\n",
            ":- import_module char, float, int, int16, int8, list, uint32, uint8.\n\n",
        ]
        for t in self.types:
            parts.append(self.declaration(t) + "\n")
        bodies = []
        for t in self.types:
            bodies.append(self.show_pred(t))
            bodies.append(self.code_func(t))
        goals = []
        for t in self.types:
            goals += self.tests(t)
        run = (
            "% N is 0, but known only at run time, so that the terms built\n"
            "% from it are allocated.\n"
            ":- pred run(int::in, io::di, io::uo) is det.\n"
            "run(N, !IO) :-\n    " + ",\n    ".join(goals) + ".\n")
        parts.append(POLYMORPHIC_CODE + "\n")
        parts += [b + "\n" for b in bodies]
        parts += [h + "\n" for h in self.helpers.values()]
        parts += [v + "\n" for v in self.values]
        parts.append(run + "\n")
        parts.append("main(!IO) :-\n    run(0, !IO).\n")
        return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("dir")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(args.seed * 100003 + i)
        name = "layouts_%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(Generator(rng).module(name))


if __name__ == "__main__":
    main()
