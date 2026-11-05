#!/usr/bin/env python3
"""Generate random programs for compare-runs.py that use type classes.

Each module has types of every representation (int, float, string, an enum, a
notag on a float, a type of cells, a type with one constant, and a polymorphic
box), type classes, some with a superclass or two parameters, whose methods
are functions, det and semidet predicates, a predicate with many outputs, one
with two modes, and one with its own type variable and constraint, and
instances of them for those types, for lists, with a constraint on the
element's type, and for boxes, with or without one. Polymorphic predicates
call methods, superclasses' methods and each other at the same type and at
types built from it, compare values and print them, call methods at instances
known from their own dictionary, and main calls them and methods at types
without type variables, where the instance is known. Existentially typed
predicates give values of types they choose, with a constraint or without, for
callers to pass on, build lists of, compare and print.

Nothing here can be chosen by mmc and merkc differently: values are printed
with io.write, and the programs avoid what merkc does not compile yet (the
library's list and string predicates, disjunctions, closures).
"""

import argparse
import os
import random

PRELUDE = """\
:- module {name}.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list.

:- type e ---> ea ; eb ; ec.
:- type n ---> n(float).
:- type t ---> t1(int, float) ; t2(string) ; t3 ; t4(e, list(int)).
:- type uu ---> uu.
:- type box(A) ---> box(A, int).
"""

# Each base type: its name, values, and an int computed from a value V, used
# in its instances' methods; the same with Delta in a value of the type.
BASES = [
    ("int", ["0", "7", "-3"], "V", "V + Delta"),
    ("float", ["0.5", "2.25"], "truncate_to_int(V)", "V + float(Delta)"),
    ("string", ['"s"', '"tt"'],
        '( if V = "s" then 1 else 2 )', '( if Delta > 0 then "s" else V )'),
    ("e", ["ea", "ec"], "e_int(V)", "( if Delta > 1 then ec else V )"),
    ("n", ["n(1.5)"], "n_int(V)", "n_add(V, Delta)"),
    ("t", ["t1(4, 0.5)", 't2("x")', "t3", "t4(eb, [1, 2])"], "t_int(V)",
        "t_add(V, Delta)"),
    ("uu", ["uu"], "1", "V"),
]

HELPERS = """
:- func e_int(e) = int.
e_int(ea) = 1.
e_int(eb) = 2.
e_int(ec) = 3.
:- func n_int(n) = int.
n_int(n(F)) = truncate_to_int(F).
:- func n_add(n, int) = n.
n_add(n(F), D) = n(F + float(D)).
:- func t_int(t) = int.
t_int(t1(I, _)) = I.
t_int(t2(_)) = 20.
t_int(t3) = 30.
t_int(t4(E, _)) = e_int(E).
:- func t_add(t, int) = t.
t_add(t1(I, F), D) = t1(I + D, F).
t_add(t2(S), _) = t2(S).
t_add(t3, _) = t3.
t_add(t4(E, L), D) = t4(E, [D | L]).
"""


class Generator:
    def __init__(self, rng, name):
        self.rng = rng
        self.name = name
        self.out = [PRELUDE.format(name=name), HELPERS]

    def emit(self, text):
        self.out.append(text)

    def typeclass(self, i, parent):
        rng = self.rng
        self.emit(f":- typeclass c{i}(T){f' <= c{parent}(T)' if parent is not None else ''} where [")
        self.emit(f"    func val{i}(T) = int,")
        self.emit(f"    pred bump{i}(int::in, T::in, T::out) is det,")
        self.emit(f"    pred big{i}(T::in, int::in) is semidet,")
        self.emit(f"    pred many{i}(T::in, T::out, int::out, float::out, T::out) is det,")
        self.emit(f"    pred two{i}(T, T),")
        self.emit(f"    mode two{i}(in, out) is det,")
        self.emit(f"    mode two{i}(out, in) is det,")
        self.emit(f"    pred other{i}(T::in, Q::in, int::out) is det <= c{i}(Q)")
        self.emit("].")

    def instance_methods(self, i, value, add, k):
        """Methods from int and new-value expressions over V and Delta."""
        return [
            f"    (val{i}(V) = {value} + {k}),",
            f"    (bump{i}(Delta, V, R) :- R = {add}),",
            f"    (big{i}(V, N) :- {value} + {k} > N),",
            f"    (many{i}(V, R1, I, F, R2) :- Delta = {k}, R1 = {add}, "
            f"I = {value}, F = float(I) / 2.0, R2 = V),",
            f"    two{i}(X, X),",
            f"    (other{i}(V, Q, R) :- R = {value} + val{i}(Q))",
        ]

    def instances(self, i, parent, kinds):
        rng = self.rng
        for base, _, value, add in BASES:
            self.emit(f":- instance c{i}({base}) where [")
            self.emit("\n".join(self.instance_methods(i, value, add, rng.randint(0, 9))))
            self.emit("].")
        # Lists, through recursive predicates that use the element's instance.
        self.emit(f":- instance c{i}(list(T)) <= c{i}(T) where [")
        self.emit("\n".join([
            f"    func(val{i}/1) is sum{i},",
            f"    pred(bump{i}/3) is bump_all{i},",
            f"    (big{i}(L, N) :- sum{i}(L) > N),",
            f"    (many{i}(L, R1, I, F, R2) :- bump_all{i}(1, L, R1), I = sum{i}(L), "
            f"F = float(I) * 0.25, R2 = L),",
            f"    two{i}(X, X),",
            f"    (other{i}(L, Q, R) :- R = sum{i}(L) * 2 + val{i}(Q))",
        ]))
        self.emit("].")
        self.emit(f":- func sum{i}(list(T)) = int <= c{i}(T).")
        self.emit(f"sum{i}([]) = 0.")
        self.emit(f"sum{i}([X | Xs]) = val{i}(X) + sum{i}(Xs).")
        self.emit(f":- pred bump_all{i}(int::in, list(T)::in, list(T)::out) is det <= c{i}(T).")
        self.emit(f"bump_all{i}(_, [], []).")
        self.emit(f"bump_all{i}(D, [X | Xs], [Y | Ys]) :- bump{i}(D, X, Y), bump_all{i}(D, Xs, Ys).")
        # Boxes: an instance with a constraint, or with a type_info of its own.
        constrained = kinds[i]
        head = f"box(A)) <= c{i}(A)" if constrained else "box(A))"
        self.emit(f":- instance c{i}({head} where [")
        value = f"(N + val{i}(A))" if constrained else "N"
        self.emit("\n".join([
            f"    (val{i}(box(A, N)) = {value}),",
            f"    (bump{i}(D, box(A, N), box(A, N + D))),",
            f"    (big{i}(box(A, N), M) :- {value} > M),",
            f"    (many{i}(box(A, N), box(A, N + 1), {value}, 0.5, box(A, N))),",
            f"    two{i}(X, X),",
            f"    (other{i}(box(A, N), Q, R) :- R = {value} + val{i}(Q))",
        ]))
        self.emit("].")

    def poly(self, i, cls, parent, earlier):
        """A polymorphic predicate that prints what the class's methods give."""
        rng = self.rng
        lines = [
            f":- pred show{i}(int::in, T::in, io::di, io::uo) is det <= c{cls}(T).",
            f"show{i}(Depth, X, !IO) :-",
            f"    io.write_int(val{cls}(X), !IO), io.write_string(\" \", !IO),",
            f"    bump{cls}(1, X, Y), io.write(Y, !IO), io.write_string(\" \", !IO),",
            f"    ( if big{cls}(X, 2) then io.write_string(\"big \", !IO)"
            f" else io.write_string(\"small \", !IO) ),",
            f"    many{cls}(X, A, I, F, B), io.write(A, !IO), io.write_string(\" \", !IO),",
            f"    io.write_int(I, !IO), io.write_string(\" \", !IO),",
            f"    io.write_float(F, !IO), io.write_string(\" \", !IO), io.write(B, !IO),",
            f"    io.write_string(\" \", !IO),",
            f"    two{cls}(X, C), two{cls}(D, C), ( if D = X then io.write_string(\"same \", !IO)"
            f" else io.write_string(\"diff \", !IO) ),",
            f"    other{cls}(X, 3, O1), other{cls}(X, 0.5, O2), other{cls}(X, [X], O3),",
            f"    io.write_int(O1 + 100 * O2 + 10000 * O3, !IO), io.write_string(\" \", !IO),",
        ]
        if parent is not None:
            lines.append(f"    io.write_int(val{parent}(X), !IO), io.write_string(\" \", !IO),")
        # Instances known here, whose premises are this predicate's dictionary
        # and whose type_infos are built from T's.
        lines.append(f"    bump{cls}(2, [X, Y], L), io.write(L, !IO), io.write_string(\" \", !IO),")
        lines.append(f"    bump{cls}(2, box(X, 1), P), io.write(P, !IO), io.write_string(\" \", !IO),")
        lines.append(f"    io.write_int(val{cls}([X]) + val{cls}(box(Y, 3)), !IO), io.write_string(\" \", !IO),")
        lines.append(f"    io.nl(!IO),")
        # Calls at the same type and at types built from it.
        calls = []
        if earlier:
            e, ecls = rng.choice(earlier)
            if ecls == cls:
                calls.append(f"show{e}(Depth - 1, {rng.choice(['X', '[X]', '[X, Y]', 'box(X, 1)'])}, !IO)")
            elif parent == ecls:
                calls.append(f"show{e}(Depth - 1, X, !IO)")
        if calls:
            lines.append(f"    ( if Depth > 0 then {', '.join(calls)} else true ),")
        lines.append(f"    show_end(!IO).")
        self.emit("\n".join(lines))

    def generate(self):
        rng = self.rng
        n = rng.randint(2, 3)
        parents = [None] + [rng.choice([None, j - 1]) for j in range(1, n)]
        # A subclass's instance for a box must hold where its superclass's
        # does, so it is constrained if its superclass's is.
        kinds = [rng.random() < 0.5 for _ in range(n)]
        for i in range(1, n):
            kinds[i] = kinds[i] or (parents[i] is not None and kinds[parents[i]])
        for i in range(n):
            self.typeclass(i, parents[i])
        for i in range(n):
            self.instances(i, parents[i], kinds)
        self.emit(":- pred show_end(io::di, io::uo) is det.")
        self.emit("show_end(!IO).")
        polys = []
        for i in range(rng.randint(2, 5)):
            cls = rng.randrange(n)
            self.poly(i, cls, parents[cls], polys)
            polys.append((i, cls))
        # Existentially typed predicates and functions: a constraint's types are
        # type constructors, since mmc mishandles a variable inside a type (the
        # design document's register).
        gives = []
        for g in range(rng.randint(2, 4)):
            cls = rng.randrange(n)
            base, values, _, _ = rng.choice(BASES)
            value = rng.choice(values)
            self.emit(f":- some [T] pred give{g}(int::in, T::out) is det => c{cls}(T).")
            self.emit(f"give{g}(_, {value} : {base}).")
            self.emit(f":- some [T] pred pass{g}(int::in, T::out) is det => c{cls}(T).")
            self.emit(f"pass{g}(N, X) :- give{g}(N, X).")
            self.emit(f":- some [T] func free{g}(int) = T.")
            self.emit(f"free{g}(_) = {value} : {base}.")
            gives.append((g, cls))
        calls = []
        for g, cls in gives:
            for kind in ["give", "pass"]:
                v = f"{kind.upper()}{g}"
                calls.append(f"{kind}{g}(1, {v})")
                calls.append(f"io.write_int(val{cls}({v}), !IO), io.write({v}, !IO), io.nl(!IO)")
                calls.append(f"bump{cls}(2, {v}, {v}B), io.write({v}B, !IO), io.nl(!IO)")
                calls.append(f"io.write_int(val{cls}([{v}, {v}B]), !IO), io.nl(!IO)")
                calls.append(f"compare(O{v}, {v}, {v}B), io.write(O{v}, !IO), io.nl(!IO)")
                for i, c in polys:
                    if c == cls and rng.random() < 0.5:
                        calls.append(f"show{i}(0, {v}, !IO)")
            calls.append(f"F{g} = free{g}(1), io.write(F{g}, !IO), io.write([F{g}], !IO), io.nl(!IO)")
        for i, cls in polys:
            for base, values, _, _ in BASES:
                if rng.random() < 0.6:
                    value = rng.choice(values)
                    calls.append(f"show{i}(1, {value}, !IO)")
                    if rng.random() < 0.3:
                        calls.append(f"show{i}(1, [{value}] : list({base}), !IO)")
                    if rng.random() < 0.3:
                        calls.append(f"show{i}(1, box({value}, 2), !IO)")
        # Methods at known instances, some with hidden arguments of their own.
        for k, (base, values, _, _) in enumerate(BASES):
            value = rng.choice(values)
            i = rng.randrange(n)
            for j, at in enumerate([value, f"[{value}, {value}]", f"box({value}, 2)", f"[box({value}, 1)]"]):
                calls.append(f"io.write_int(val{i}({at}), !IO)")
                calls.append(f"bump{i}(3, {at}, M{k}_{j})")
                calls.append(f"io.write(M{k}_{j}, !IO)")
                calls.append(f"many{i}({at}, A{k}_{j}, B{k}_{j}, C{k}_{j}, D{k}_{j})")
                calls.append(f"io.write(A{k}_{j}, !IO), io.write_int(B{k}_{j}, !IO), "
                             f"io.write_float(C{k}_{j}, !IO), io.write(D{k}_{j}, !IO)")
                calls.append("io.nl(!IO)")
        self.emit("main(!IO) :-\n    " + ",\n    ".join(calls) + ".")
        return "\n".join(self.out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("dir")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        name = f"classes_{args.seed}_{i}"
        rng = random.Random(f"{args.seed}/{i}")
        text = Generator(rng, name).generate()
        with open(os.path.join(args.dir, name + ".m"), "w") as out:
            out.write(text)


if __name__ == "__main__":
    main()
