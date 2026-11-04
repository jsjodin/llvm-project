#!/usr/bin/env python3
"""Generate random programs for compare-runs.py that compute and print.

Each module has a few procedures of each kind, each of which may call the
earlier ones:

- int functions of two arguments, whose bodies are expressions of
  arithmetic, division and remainders by constants that are not zero, bit
  operations, shifts by small constants, min and max, if-then-else and
  calls;
- det predicates of two outputs, set in the branches of an if-then-else;
- semidet predicates: comparisons, conjunctions, negations and calls;
- semidet predicates that switch on an int, and are called in conditions,
  where their output is used;
- recursive functions, both tail recursive, with an accumulator, and not;
- float functions, printed with io.write_float;
- for the types of generate-layout-programs.py, a function that builds a
  value from an int and one that folds a value into an int, by clauses.

Values are kept small, each function's result taken modulo a prime and
multiplications by small constants only, so that no int overflows: the
language leaves overflow to the implementation. main calls each procedure
on fixed arguments and, in a loop, on arguments computed at run time, and
prints the results. The modules are det and print only, so that their
output is the comparison.

Usage: generate-run-programs.py [--seed N] [--count N] DIR
"""

import argparse
import importlib.util
import os
import random

_spec = importlib.util.spec_from_file_location(
    "layouts", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "generate-layout-programs.py"))
layouts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layouts)

PRIME = 100003
MAX_DEPTH = 3


def literal(n):
    return str(n) if n >= 0 else "(%d)" % n


def nonzero(rng):
    return rng.choice([-7, -3, -2, -1, 1, 2, 3, 5, 9])


class Generator:
    def __init__(self, rng):
        self.rng = rng
        self.functions = []    # names of int functions of two ints
        self.pairs = []        # names of det predicates (int, int, out, out)
        self.tests = []        # names of semidet predicates (int, int)
        self.switches = []     # (name, cases) semidet (int::in, int::out)
        self.loops = []        # names of recursive functions (int, int) = int
        self.floats = []       # names of float functions of an int
        self.code = []
        self.layouts = layouts.Generator(rng)
        self.terms = []        # types with make_ and sum_ functions
        self.folds = {}        # key -> code of a fold helper

    # Expressions over int variables.

    def expr(self, variables, depth=0):
        rng = self.rng
        if depth >= MAX_DEPTH or rng.random() < 0.25:
            if variables and rng.random() < 0.7:
                return rng.choice(variables)
            return literal(rng.randint(-20, 20))
        sub = lambda: self.expr(variables, depth + 1)
        choice = rng.choice(
            ["add", "sub", "mul", "div", "rem", "divf", "mod", "and", "or",
             "xor", "not", "shl", "shr", "min", "max", "ite", "call",
             "call", "pair", "switch", "loop"])
        if choice == "add":
            return "(%s + %s)" % (sub(), sub())
        if choice == "sub":
            return "(%s - %s)" % (sub(), sub())
        if choice == "mul":
            return "(%s * %s)" % (sub(), literal(rng.randint(-9, 9)))
        if choice in ("div", "rem", "divf", "mod"):
            op = {"div": "//", "rem": "rem", "divf": "div", "mod": "mod"}[choice]
            return "(%s %s %s)" % (sub(), op, literal(nonzero(rng)))
        if choice == "and":
            return "(%s /\\ %s)" % (sub(), sub())
        if choice == "or":
            return "(%s \\/ %s)" % (sub(), sub())
        if choice == "xor":
            return "xor(%s, %s)" % (sub(), sub())
        if choice == "not":
            return "(\\ %s)" % sub()
        if choice in ("shl", "shr"):
            op = "<<" if choice == "shl" else ">>"
            return "(%s %s %d)" % (sub(), op, rng.randint(0, 6))
        if choice in ("min", "max"):
            # Through functions of the module's own, since int also has the
            # predicates min/3 and max/3, whose closures make
            # min(A, B) = max(C, D) ambiguous. A type annotation would
            # resolve it, but makes the annotated call output to the other
            # side's variable, which needs the reverse modes of int's
            # operations in the calls under it.
            return "i%s(%s, %s)" % (choice, sub(), sub())
        if choice == "ite":
            return "( if %s then %s else %s )" % (
                self.cond(variables, depth + 1), sub(), sub())
        if choice == "call" and self.functions:
            return "%s(%s, %s)" % (rng.choice(self.functions), sub(), sub())
        if choice == "loop" and self.loops:
            # A loop's count is kept small.
            return "%s(%s mod 17, %s)" % (rng.choice(self.loops), sub(), sub())
        if choice == "pair" and self.pairs:
            # A det predicate's outputs, through a conditional so that they
            # are an expression; they are bound only in its then part.
            x, y = self.var("P"), self.var("Q")
            return "( if %s(%s, %s, %s, %s), %s > %s then %s - %s else %s )" % (
                rng.choice(self.pairs), sub(), sub(), x, y, x, y, x, y, sub())
        if choice == "switch" and self.switches:
            name, _ = rng.choice(self.switches)
            r = self.var("R")
            return "( if %s(%s, %s) then %s else %s )" % (
                name, sub(), r, r, sub())
        return "(%s + %s)" % (sub(), sub())

    def cond(self, variables, depth=0):
        rng = self.rng
        sub = lambda: self.expr(variables, depth + 1)
        if depth >= MAX_DEPTH:
            choice = "compare"
        else:
            choice = rng.choice(["compare", "compare", "compare", "and",
                                 "not", "test", "switch"])
        if choice == "and":
            return "(%s, %s)" % (self.cond(variables, depth + 1),
                                 self.cond(variables, depth + 1))
        if choice == "not":
            # Of a comparison: the negation of a conjunction of negations,
            # such as \+ (A \= 1, B \= 2), is a disjunction, as in mmc,
            # which code generation does not handle yet.
            op = rng.choice(["<", ">", "=<", ">=", "="])
            return "\\+ %s %s %s" % (sub(), op, sub())
        if choice == "test" and self.tests:
            return "%s(%s, %s)" % (rng.choice(self.tests), sub(), sub())
        if choice == "switch" and self.switches:
            name, _ = rng.choice(self.switches)
            r = self.var("R")
            return "(%s(%s, %s), %s %s %s)" % (
                name, sub(), r, r, rng.choice(["<", ">", "=", "\\="]), sub())
        op = rng.choice(["<", ">", "=<", ">=", "=", "\\="])
        return "%s %s %s" % (sub(), op, sub())

    def var(self, prefix):
        self.layouts.fresh += 1
        return "%s%d" % (prefix, self.layouts.fresh)

    # Procedures.

    def function(self):
        name = "f%d" % len(self.code)
        body = self.expr(["A", "B"])
        self.code.append(
            ":- func %s(int, int) = int.\n%s(A, B) = (%s) rem %d.\n"
            % (name, name, body, PRIME))
        self.functions.append(name)

    def pair(self):
        name = "p%d" % len(self.code)
        v = ["A", "B"]
        self.code.append(
            ":- pred %s(int::in, int::in, int::out, int::out) is det.\n"
            "%s(A, B, X, Y) :-\n"
            "    ( if %s then\n"
            "        X = (%s) rem %d,\n"
            "        Y = (%s) rem %d\n"
            "    else\n"
            "        Y = (%s) rem %d,\n"
            "        X = (%s) rem %d\n"
            "    ).\n" % (name, name, self.cond(v), self.expr(v), PRIME,
                          self.expr(v), PRIME, self.expr(v), PRIME,
                          self.expr(v), PRIME))
        self.pairs.append(name)

    def test(self):
        name = "s%d" % len(self.code)
        self.code.append(":- pred %s(int::in, int::in) is semidet.\n%s(A, B) :-\n    %s.\n"
                         % (name, name, self.cond(["A", "B"])))
        self.tests.append(name)

    def switch(self):
        rng = self.rng
        name = "w%d" % len(self.code)
        cases = sorted(rng.sample(range(-4, 12), rng.randint(1, 6)))
        arms = ["X = %s, R = (%s) rem %d" % (literal(c), self.expr(["X"]), PRIME)
                for c in cases]
        self.code.append(
            ":- pred %s(int::in, int::out) is semidet.\n%s(X, R) :-\n"
            "    (\n        %s\n    ).\n" % (name, name, "\n    ;\n        ".join(arms)))
        self.switches.append((name, cases))

    def loop(self):
        rng = self.rng
        name = "l%d" % len(self.code)
        step = self.expr(["N", "Acc"], 1)
        if rng.random() < 0.5:
            # Tail recursive, with an accumulator.
            self.code.append(
                ":- func %s(int, int) = int.\n"
                "%s(N, Acc) =\n"
                "    ( if N =< 0 then\n        Acc\n    else\n"
                "        %s(N - 1, (Acc * %d + %s) rem %d)\n    ).\n"
                % (name, name, name, rng.randint(-3, 3), step, PRIME))
        else:
            # Its result is combined after the recursive call.
            self.code.append(
                ":- func %s(int, int) = int.\n"
                "%s(N, Acc) =\n"
                "    ( if N =< 0 then\n        Acc\n    else\n"
                "        (%s + %s(N - 1, Acc) * %d) rem %d\n    ).\n"
                % (name, name, step, name, rng.randint(-3, 3), PRIME))
        self.loops.append(name)

    def fexpr(self, depth=0):
        rng = self.rng
        if depth >= MAX_DEPTH or rng.random() < 0.3:
            return rng.choice(["F", "F", "%s" % repr(rng.randint(-8, 8) + rng.choice([0.0, 0.5, 0.25]))])
        sub = lambda: self.fexpr(depth + 1)
        choice = rng.choice(["add", "sub", "mul", "div", "ite", "int"])
        if choice == "add":
            return "(%s + %s)" % (sub(), sub())
        if choice == "sub":
            return "(%s - %s)" % (sub(), sub())
        if choice == "mul":
            return "(%s * %s)" % (sub(), repr(rng.choice([-1.5, 0.5, 2.0, 3.0, 0.1])))
        if choice == "div":
            return "(%s / %s)" % (sub(), repr(rng.choice([-4.0, 3.0, 7.0, 0.5])))
        if choice == "int":
            return "float(float.truncate_to_int(%s) rem %d)" % (sub(), PRIME)
        return "( if %s %s %s then %s else %s )" % (
            sub(), rng.choice(["<", ">", "=<", ">="]), sub(), sub(), sub())

    def float_function(self):
        name = "g%d" % len(self.code)
        self.code.append(
            ":- func %s(int) = float.\n%s(A) = R :-\n    F = float(A),\n"
            "    R = %s.\n" % (name, name, self.fexpr()))
        self.floats.append(name)

    # Terms.

    def fold(self, field, variable):
        kind = field[0]
        if kind == "int":
            return variable
        if kind == "float":
            return "(float.truncate_to_int(%s) rem %d)" % (variable, PRIME)
        if kind == "char":
            return "( if %s = '%s' then 7 else 1 )" % (
                variable, self.rng.choice("abcdefghijklmnopqrstuvwxyz"))
        if kind == "type":
            return "sum_%s(%s)" % (field[1].name, variable)
        key = layouts.key(field)
        if key not in self.folds:
            self.folds[key] = None
            mtype = layouts.type_string(field)
            if kind == "list":
                code = (":- func sum_{n}({t}) = int.\n"
                        "sum_{n}([]) = 0.\n"
                        "sum_{n}([X | Xs]) = ({e} * 3 + sum_{n}(Xs)) rem {p}.\n"
                        ).format(n=key, t=mtype, e=self.fold(field[1], "X"), p=PRIME)
            else:
                parts = ["A%d" % i for i in range(len(field) - 1)]
                folds = [self.fold(f, a) for f, a in zip(field[1:], parts)]
                code = (":- func sum_{n}({t}) = int.\n"
                        "sum_{n}({{{a}}}) = ({e}) rem {p}.\n"
                        ).format(n=key, t=mtype, a=", ".join(parts),
                                 e=" + ".join(folds), p=PRIME)
            self.folds[key] = code
        return "sum_%s(%s)" % (key, variable)

    def term_functions(self, t):
        name = t.name
        lines = [":- func sum_%s(%s) = int." % (name, name)]
        for i, (functor, fields) in enumerate(t.constructors):
            arguments = ["A%d" % j for j in range(len(fields))]
            pattern = functor if not fields else "%s(%s)" % (functor, ", ".join(arguments))
            folds = [self.fold(f, a) for f, a in zip(fields, arguments)]
            body = " + ".join(["%d" % (i * 11)] + folds)
            lines.append("sum_%s(%s) = (%s) rem %d." % (name, pattern, body, PRIME))
        self.code.append("\n".join(lines) + "\n")
        # make_t(N) chooses among a few values by N, with N in them.
        values = [self.layouts.value(("type", t), 0) for _ in range(3)]
        arms = [self.layouts.render(v, True) for v in values]
        self.code.append(
            ":- func make_%s(int) = %s.\n"
            "make_%s(N) =\n"
            "    ( if N mod 3 = 0 then\n        %s\n"
            "    else if N mod 3 = 1 then\n        %s\n"
            "    else\n        %s\n    ).\n"
            % (name, name, name, arms[0], arms[1], arms[2]))
        self.terms.append(t)

    # The module.

    def module(self, name):
        rng = self.rng
        kinds = ([self.function] * 4 + [self.pair] * 2 + [self.test] * 2 +
                 [self.switch] * 2 + [self.loop] * 2 + [self.float_function])
        for _ in range(rng.randint(6, 12)):
            rng.choice(kinds)()
        for i in range(rng.randint(1, 3)):
            t = self.layouts.make_type(i + 1)
        for t in self.layouts.types:
            self.term_functions(t)

        goals = []
        drives = []

        def show(call, kind="int"):
            write = "io.write_float" if kind == "float" else "io.write_int"
            return ["io.write_string(\"%s = \", !IO)" % call.replace('"', '\\"'),
                    "%s(%s, !IO)" % (write, call), "io.nl(!IO)"]

        def fixed():
            return literal(rng.randint(-30, 30))

        for f in self.functions:
            for _ in range(2):
                goals += show("%s(%s, %s)" % (f, fixed(), fixed()))
            drives.append(["io.write_int(%s(I, I * %d - %d), !IO)" % (
                f, rng.randint(-5, 5), rng.randint(0, 9))])
        for p in self.pairs:
            a, b = fixed(), fixed()
            x, y = self.var("X"), self.var("Y")
            goals += ["%s(%s, %s, %s, %s)" % (p, a, b, x, y),
                      "io.write_string(\"%s(%s, %s) = \", !IO)" % (p, a, b),
                      "io.write_int(%s, !IO)" % x, "io.write_char(' ', !IO)",
                      "io.write_int(%s, !IO)" % y, "io.nl(!IO)"]
            x, y = self.var("X"), self.var("Y")
            drives.append(["%s(I, %s, %s, %s)" % (p, fixed(), x, y),
                           "io.write_int(%s - %s, !IO)" % (x, y)])
        for s in self.tests:
            a, b = fixed(), fixed()
            goals += ["( if %s(%s, %s) then io.write_string(\"%s yes\", !IO) "
                      "else io.write_string(\"%s no\", !IO) )" % (s, a, b, s, s),
                      "io.nl(!IO)"]
            drives.append(["( if %s(I, %s) then io.write_char('y', !IO) "
                           "else io.write_char('n', !IO) )" % (s, fixed())])
        for w, cases in self.switches:
            r = self.var("R")
            drives.append(["( if %s(I, %s) then io.write_int(%s, !IO) "
                           "else io.write_char('-', !IO) )" % (w, r, r)])
        for l in self.loops:
            goals += show("%s(%d, %s)" % (l, rng.randint(0, 40), fixed()))
            drives.append(["io.write_int(%s(I + 5, I), !IO)" % l])
        for g in self.floats:
            goals += show("%s(%s)" % (g, fixed()), "float")
            drives.append(["io.write_float(%s(I), !IO)" % g])
        for t in self.terms:
            drives.append(["show_%s(make_%s(I), !IO)" % (t.name, t.name),
                           "io.write_char(' ', !IO)",
                           "io.write_int(sum_%s(make_%s(I)), !IO)" % (t.name, t.name)])

        low, high = -rng.randint(2, 8), rng.randint(3, 12)
        drive_code = []
        for i, drive in enumerate(drives):
            drive_code.append(
                ":- pred drive%d(int::in, io::di, io::uo) is det.\n"
                "drive%d(I, !IO) :-\n"
                "    ( if I > %d then\n        true\n    else\n"
                "        io.write_string(\"drive%d \", !IO),\n"
                "        io.write_int(I, !IO),\n"
                "        io.write_string(\": \", !IO),\n"
                "        %s,\n"
                "        io.nl(!IO),\n"
                "        drive%d(I + 1, !IO)\n    ).\n"
                % (i, i, high, i, ",\n        ".join(drive), i))
            goals.append("drive%d(%d, !IO)" % (i, low))

        layout_bodies = []
        for t in self.layouts.types:
            layout_bodies.append(self.layouts.show_pred(t))
        parts = [
            ":- module %s.\n" % name,
            ":- interface.\n",
            ":- import_module io.\n",
            ":- pred main(io::di, io::uo) is det.\n",
            ":- implementation.\n",
            ":- import_module char, float, int, list.\n\n",
            ":- func imin(int, int) = int.\nimin(A, B) = min(A, B).\n\n",
            ":- func imax(int, int) = int.\nimax(A, B) = max(A, B).\n\n",
        ]
        parts += [self.layouts.declaration(t) + "\n" for t in self.layouts.types]
        parts += [c + "\n" for c in self.code]
        parts += [c + "\n" for c in self.folds.values()]
        parts += [b + "\n" for b in layout_bodies]
        parts += [h + "\n" for h in self.layouts.helpers.values()]
        parts += [d + "\n" for d in drive_code]
        parts.append("main(!IO) :-\n    %s.\n" % ",\n    ".join(goals))
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
        rng = random.Random(args.seed * 100019 + i)
        name = "runs_%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(Generator(rng).module(name))


if __name__ == "__main__":
    main()
