#!/usr/bin/env python3
"""Generate random modules for compare-switches.py.

Each module has predicates whose bodies are disjunctions, and clauses
whose heads are patterns, that test their inputs against functors: of a
type with constants and functors with arguments, of a type with one
functor, of lists and of integers. A disjunct may test an input through
another variable unified with it, after constructions and other tests,
inside a `some` scope or a nested disjunction, after a call, which keeps
it from being a switch, or not at all; it may test the same functor as
another, and test the arguments it binds in a disjunction after. Some
disjunctions are in if-then-elses, negations and lambda expressions, and
some inputs have bound insts. The modules are mode-correct, and declared
nondet, so that mmc checks them up to determinism analysis.

With --determinism, for compare-switches.py --determinism, the goals also
call predicates of each determinism, some with no outputs, and each
predicate and lambda expression is declared with a determinism chosen at
random, so that some declarations are not satisfied. With --committed too,
the predicates called and the determinisms declared include committed
choice ones, so that some committed choice calls are in contexts that
require all solutions. With --casts too, some disjunctions are in
promise_equivalent_solutions scopes, which list the variable they bind,
or now and then none or one more; some calls are in arbitrary scopes, and
some bodies in promise_equivalent_solution_sets scopes, so that some
arbitrary scopes are outside them. With --requires too, some disjunctions
are in require_<det>, require_complete_switch and require_switch_arms_<det>
scopes, on the variable they test or now and then another, and some calls
in require_<det> scopes, each with a determinism chosen at random, so that
some requirements are not met. With --inferred too, some predicates are
declared without a determinism, in the implementation section, and the
goals also call predicates so declared, whose bodies call the others and
each other, recursively or not, so that their determinisms are inferred as
a fixpoint.

Usage: generate-switch-programs.py [--seed N] [--count N] [--determinism
                                   [--committed] [--casts] [--requires]
                                   [--inferred]] DIR
"""

import argparse
import os
import random

PRELUDE = """\
:- module {name}.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c(int) ; d(t, int).
:- type u ---> u(t, int).
:- inst ab == bound(a ; b).
:- inst abc == bound(a ; b ; c(ground)).

"""

INPUTS = [("X", "t"), ("W", "t"), ("L", "list(int)"), ("U", "u"), ("N", "int")]

# Predicates of each determinism, for --determinism.
HELPERS = """\
:- pred sd(int::in, int::out) is semidet.
sd(X, X) :- X > 0.
:- pred md(int::in, int::out) is multi.
md(X, Y) :- ( Y = X ; Y = X + 1 ).
:- pred nd(int::in, int::out) is nondet.
nd(X, Y) :- X > 0, ( Y = X ; Y = 1 ).
:- pred fl(int::in, int::out) is failure.
fl(_, _) :- fail.
:- pred er(int::in, int::out) is erroneous.
er(_, _) :- error("er").

"""
CALLS = ["q", "q", "sd", "md", "nd", "fl", "er"]
DETERMINISMS = ["det", "det", "semidet", "semidet", "multi", "nondet", "nondet",
                "failure", "erroneous"]
# Committed choice ones, for --committed.
COMMITTED_HELPERS = """\
:- pred cm(int::in, int::out) is cc_multi.
cm(X, Y) :- ( Y = X ; Y = X + 1 ).
:- pred cn(int::in, int::out) is cc_nondet.
cn(X, Y) :- X > 0, ( Y = X ; Y = 1 ).

"""
COMMITTED_CALLS = ["cm", "cm", "cn"]
COMMITTED_DETERMINISMS = ["cc_multi", "cc_multi", "cc_nondet"]
determinism = False
casts = False
requires = False
inferred = False
committed = False
# The bodies of the predicates declared without a determinism, for
# --inferred, with C and D for callees.
INFERRED_BODIES = [
    "C(X, Y)",
    "C(X, Z), D(Z, Y)",
    "( X > 0, C(X - 1, Y) ; Y = X )",
    "( if X = 0 then Y = 0 else C(X - 1, Z), D(Z, Y) )",
    "( C(X, Y) ; D(X, Y) )",
    "( if C(X, Z) then D(Z, Y) else Y = X )",
    "X > 2, C(X, Y)",
]


class Gen:
    def __init__(self, rng, calls=None):
        self.rng = rng
        self.count = 0
        self.calls = calls or CALLS

    def fresh(self, base="V"):
        self.count += 1
        return "%s%d" % (base, self.count)

    def int_value(self, ctx):
        ints = [v for v, t in ctx if t == "int"]
        if ints and self.rng.random() < 0.5:
            return self.rng.choice(ints)
        return str(self.rng.randint(0, 3))

    def test(self, var, typ, ctx, goals):
        """A test of var against a functor of its type; the variables it
        binds are added to ctx."""
        r = self.rng
        if typ == "t":
            f = r.choice(["a", "b", "c", "c", "d"])
            if f in ("a", "b"):
                goals.append("%s = %s" % (var, f))
            elif f == "c":
                if r.random() < 0.3:
                    goals.append("%s = c(%s)" % (var, self.int_value(ctx)))
                else:
                    k = self.fresh("K")
                    goals.append("%s = c(%s)" % (var, k))
                    ctx.append((k, "int"))
            else:
                s, k = self.fresh("S"), self.fresh("K")
                goals.append("%s = d(%s, %s)" % (var, s, k))
                ctx.append((s, "t"))
                ctx.append((k, "int"))
        elif typ == "list(int)":
            f = r.choice(["nil", "cons", "cons", "one", "two"])
            if f == "nil":
                goals.append("%s = []" % var)
            elif f == "one":
                h = self.fresh("H")
                goals.append("%s = [%s]" % (var, h))
                ctx.append((h, "int"))
            elif f == "two":
                goals.append("%s = [_, _ | _]" % var)
            else:
                h, tl = self.fresh("H"), self.fresh("T")
                goals.append("%s = [%s | %s]" % (var, h, tl))
                ctx.append((h, "int"))
                ctx.append((tl, "list(int)"))
        elif typ == "u":
            s, k = self.fresh("S"), self.fresh("K")
            goals.append("%s = u(%s, %s)" % (var, s, k))
            ctx.append((s, "t"))
            ctx.append((k, "int"))
        else:
            goals.append("%s = %d" % (var, r.randint(0, 3)))

    def prefix(self, ctx, goals):
        r = self.rng
        for _ in range(r.choice([0, 0, 1, 1, 2])):
            k = r.random()
            if k < 0.35:
                v, t = r.choice([(v, t) for v, t in ctx if t != "int"] or ctx)
                z = self.fresh("Z")
                goals.append(r.choice(["%s = %s", "%s = %s"]) % (z, v) if r.random() < 0.7
                             else "%s = %s" % (v, z))
                ctx.append((z, t))
            elif k < 0.6:
                m = self.fresh("M")
                goals.append("%s = %d" % (m, r.randint(0, 3)))
                ctx.append((m, "int"))
            elif k < 0.75:
                m = self.fresh("M")
                callee = r.choice(self.calls) if determinism else "q"
                if determinism and r.random() < 0.2:
                    # No outputs.
                    goals.append("%s(%s, _)" % (callee, self.int_value(ctx)))
                else:
                    call = "%s(%s, %s)" % (callee, self.int_value(ctx), m)
                    if casts and r.random() < 0.15:
                        call = "arbitrary [%s] %s" % (m, call)
                    if requires and r.random() < 0.1:
                        call = "require_%s (%s)" % (r.choice(DETERMINISMS), call)
                    goals.append(call)
                    ctx.append((m, "int"))
            elif k < 0.85:
                goals.append("%s > 0" % self.int_value(ctx))
            else:
                v, t = r.choice(ctx)
                self.test(v, t, ctx, goals)

    def disjunct(self, ctx, out, depth, target):
        """A conjunction that binds `out`, testing `target` maybe."""
        r = self.rng
        ctx = list(ctx)
        goals = []
        self.prefix(ctx, goals)
        k = r.random()
        if k < 0.65:
            var, typ = target
            aliases = [v for v, t in ctx if t == typ]
            if r.random() < 0.3 and aliases:
                var = r.choice(aliases)
            test = []
            self.test(var, typ, ctx, test)
            if r.random() < 0.15:
                goals.append("some [%s] (%s)" % (self.fresh("Q"), ", ".join(test)))
            else:
                goals.extend(test)
        elif k < 0.8 and depth < 2:
            # A nested disjunction, whose branches bind a variable that is
            # used after it.
            v = self.fresh("R")
            branches = [self.disjunct(ctx, v, depth + 1, target)
                        for _ in range(r.choice([2, 2, 3]))]
            goals.append("( " + " ; ".join(branches) + " )")
            ctx.append((v, "int"))
        elif k < 0.85:
            v, t = r.choice(ctx)
            self.test(v, t, ctx, goals)
        # Then a test of what it bound, maybe in a disjunction.
        bound = [(v, t) for v, t in ctx if v[0] in "SHTK" and t != "int"]
        if bound and r.random() < 0.3 and depth < 2:
            sub = r.choice(bound)
            v = self.fresh("R")
            branches = [self.disjunct(ctx, v, depth + 1, sub) for _ in range(2)]
            goals.append("( " + " ; ".join(branches) + " )")
            ctx.append((v, "int"))
        k = r.random()
        if k < 0.1:
            goals.append("%s > 0" % self.int_value(ctx))
        elif k < 0.15:
            goals.append("not %s = %d" % (self.int_value(ctx), r.randint(0, 3)))
        goals.append("%s = %s" % (out, self.int_value(ctx)))
        return ", ".join(goals)

    def disjunction(self, ctx, out, target, depth=0):
        r = self.rng
        n = r.choice([2, 3, 3, 4, 5])
        disj = "( " + " ; ".join(self.disjunct(ctx, out, depth, target)
                                 for _ in range(n)) + " )"
        if casts and r.random() < 0.2:
            k = r.random()
            listed = [] if k < 0.1 else [out, r.choice(ctx)[0]] if k < 0.2 else [out]
            disj = "promise_equivalent_solutions [%s] %s" % (", ".join(listed), disj)
        if requires:
            k = r.random()
            var = target[0] if r.random() < 0.85 else r.choice(ctx)[0]
            if k < 0.1:
                disj = "require_%s (%s)" % (r.choice(DETERMINISMS), disj)
            elif k < 0.2:
                disj = "require_complete_switch [%s] (%s)" % (var, disj)
            elif k < 0.3:
                disj = "require_switch_arms_%s [%s] (%s)" % (
                    r.choice(DETERMINISMS), var, disj)
        return disj

    def body(self, ctx, out):
        goal = self.body_goal(ctx, out)
        if casts and self.rng.random() < 0.3:
            return "promise_equivalent_solution_sets [%s] (%s)" % (out, goal)
        return goal

    def body_goal(self, ctx, out):
        r = self.rng
        others = [c for c in ctx if c[1] != "int"]
        target = r.choice(others if others and r.random() < 0.85 else ctx)
        k = r.random()
        if k < 0.65:
            return self.disjunction(ctx, out, target)
        if k < 0.75:
            c = self.fresh("R")
            cond = self.disjunction(ctx, c, target)
            return "( if %s, %s > 0 then %s = 1 else %s )" % (
                cond, c, out, self.disjunction(ctx, out, target))
        if k < 0.85:
            c = self.fresh("R")
            return "not (%s, %s = 2), %s" % (self.disjunction(ctx, c, target), c,
                                           self.disjunction(ctx, out, target))
        p, y = self.fresh("P"), self.fresh("Y")
        detism = r.choice(DETERMINISMS) if determinism else "nondet"
        return "%s = (pred(%s::out) is %s :- %s), %s(%s)" % (
            p, y, detism, self.disjunction(ctx, y, target), p, out)


def head_pattern(gen, var, typ, ctx):
    r = gen.rng
    if r.random() < 0.3:
        return var
    goals = []
    gen.test(var, typ, ctx, goals)
    return goals[0].split(" = ", 1)[1]


def predicate(gen, n):
    r = gen.rng
    inputs = [i for i in INPUTS if r.random() < 0.6] or [INPUTS[0]]
    insts = []
    for v, t in inputs:
        if t == "t" and r.random() < 0.15:
            insts.append("%s::in(%s)" % (t, r.choice(["ab", "abc"])))
        else:
            insts.append("%s::in" % t)
    name = "p%d" % n
    detism = r.choice(DETERMINISMS) if determinism else "nondet"
    undeclared = inferred and r.random() < 0.3
    decl = ":- pred %s(%s, int::out)%s.\n" % (
        name, ", ".join(insts), "" if undeclared else " is " + detism)
    clauses = []
    if r.random() < 0.3:
        # Clauses whose heads are patterns.
        for _ in range(r.choice([2, 3, 4])):
            ctx = []
            args = []
            for v, t in inputs:
                fresh = gen.fresh(v)
                ctx.append((fresh, t))
                pattern = head_pattern(gen, fresh, t, ctx)
                if pattern != fresh:
                    ctx.remove((fresh, t))
                args.append(pattern)
            out = gen.fresh("O")
            if r.random() < 0.5:
                clauses.append("%s(%s, %s) :- %s = %s." %
                               (name, ", ".join(args), out, out, gen.int_value(ctx)))
            else:
                clauses.append("%s(%s, %s) :- %s." %
                               (name, ", ".join(args), out, gen.body(ctx, out) if ctx
                                else "%s = 1" % out))
    else:
        ctx = list(inputs)
        clauses.append("%s(%s, R) :- %s." %
                       (name, ", ".join(v for v, _ in inputs), gen.body(ctx, "R")))
    return decl, clauses, undeclared


def module(rng, name):
    # For --inferred, predicates declared without a determinism that the
    # goals call, and whose bodies call the others and each other.
    inferred_preds = []
    if inferred:
        names = ["w%d" % i for i in range(rng.randint(1, 3))]
        for w in names:
            body = rng.choice(INFERRED_BODIES)
            body = body.replace("C", rng.choice(CALLS + names)).replace(
                "D", rng.choice(CALLS + names))
            inferred_preds.append(":- pred %s(int::in, int::out).\n%s(X, Y) :- %s.\n" %
                                  (w, w, body))
        gen = Gen(rng, CALLS + names * 2)
    else:
        gen = Gen(rng)
    decls, local, defs = [], [], []
    for n in range(rng.randint(2, 5)):
        d, c, undeclared = predicate(gen, n)
        (local if undeclared else decls).append(d)
        defs.append("\n".join(c) + "\n")
    return (PRELUDE.format(name=name) + "".join(decls) +
            "\n:- implementation.\n:- import_module int%s.\n\n" % (", require" if determinism else "") +
            ":- pred q(int::in, int::out) is det.\nq(X, X).\n" +
            (HELPERS if determinism else "\n") +
            (COMMITTED_HELPERS if committed else "") + "".join(inferred_preds) +
            "".join(local) + "\n".join(defs))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--determinism", action="store_true")
    parser.add_argument("--committed", action="store_true")
    parser.add_argument("--casts", action="store_true")
    parser.add_argument("--requires", action="store_true")
    parser.add_argument("--inferred", action="store_true")
    parser.add_argument("dir")
    args = parser.parse_args()
    global determinism, casts, requires, inferred, committed
    determinism = args.determinism
    casts = args.casts
    requires = args.requires
    inferred = args.inferred
    committed = args.committed
    if args.committed:
        CALLS.extend(COMMITTED_CALLS)
        DETERMINISMS.extend(COMMITTED_DETERMINISMS)
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(args.seed * 100003 + i)
        name = "switches_%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(module(rng, name))


if __name__ == "__main__":
    main()
