#!/usr/bin/env python3
"""Generate random modules for compare-switches.py --determinism that
check unique modes where backtracking can happen.

Each predicate threads a unique value of type t, from its first argument
to its second, through updates that clobber it (`di`, `uo`) or keep it
for backtracking (`mdi`, `muo`), constructions and assignments, and
calls of the module's other predicates, between goals that can succeed
more than once or fail: calls of each determinism, disjunctions,
if-then-elses, negations and promise_equivalent_solutions scopes. The
branches of a disjunction or an if-then-else each go on from the same
value, and a condition may update the value that the else part then
uses, so that some values are needed on backtracking. Now and then a
value is used after it was clobbered, or passed where it is shared. The
predicates' modes and determinisms are chosen at random, and some are
declared without a determinism, so that it is inferred. Each goal of a
body is on a line of its own; a branched goal is on one line, since mmc
puts the notes about a disjunction in a conjunction at the line of its
semicolon.

With --scopes, a goal may also be a lambda expression that goes on from
the value, of a random determinism, called at once, a closure of a
helper, an arbitrary scope inside a promise_equivalent_solution_sets
scope, a require_<det> scope, or an `all` goal.

With --io, the value is the I/O state: t is io.state, the helpers that
update it write or keep it, and a construction of a new value is an
assignment, so that a predicate with a determinism other than det,
cc_multi or erroneous has I/O state arguments of an invalid determinism.

With --ui, the helper that reads the value takes it as `ui`, so that it
must be unique there, and not shared after.

Usage: generate-unique-programs.py [--seed N] [--count N] [--scopes] [--io]
                                   [--ui] DIR
"""

import argparse
import os
import random
import re

PRELUDE = """\
:- module {name}.
:- interface.

:- type t ---> t(int).

"""

HELPERS = """\
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred nd(int::in, int::out) is nondet.
nd(X, Y) :- X > 0, ( Y = X ; Y = 1 ).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).
:- pred sd(int::in) is semidet.
sd(X) :- X > 0.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, t(X), t(X + Y)).
:- pred upd(int::in, t::mdi, t::muo) is det.
upd(Y, t(X), t(X + Y)).
:- pred peek(t::in, int::out) is det.
peek(t(X), X).
"""

IO_PRELUDE = """\
:- module {name}.
:- interface.
:- import_module io.

:- type t == io.state.

"""

IO_HELPERS = """\
:- pred m(int::out) is multi.
m(1).
m(2).
:- pred nd(int::in, int::out) is nondet.
nd(X, Y) :- X > 0, ( Y = X ; Y = 1 ).
:- pred cm(int::out) is cc_multi.
cm(1).
cm(2).
:- pred sd(int::in) is semidet.
sd(X) :- X > 0.
:- pred updu(int::in, t::di, t::uo) is det.
updu(Y, !IO) :- io.write_int(Y, !IO).
:- pred upd(int::in, t::mdi, t::muo) is det.
upd(_, IO, IO).
:- pred peek(t::in, int::out) is det.
peek(_, 1).
"""

IO_SCOPE_HELPERS = """\
:- pred mu(t::di, t::uo) is multi.
mu(IO, IO).
mu(IO0, IO) :- io.write_int(1, IO0, IO).
:- pred mmu(t::mdi, t::muo) is multi.
mmu(IO, IO).
mmu(IO, IO).
"""

SCOPE_HELPERS = """\
:- pred mu(t::di, t::uo) is multi.
mu(T, T).
mu(t(X), t(X + 1)).
:- pred mmu(t::mdi, t::muo) is multi.
mmu(T, T).
mmu(t(X), t(X + 1)).
"""

LAMBDA_DETERMINISMS = ["det", "semidet", "multi", "nondet", "cc_multi",
                       "cc_nondet"]

MODES = [("di", "uo"), ("di", "uo"), ("di", "uo"), ("mdi", "muo"),
         ("di", "muo"), ("mdi", "uo")]
DETERMINISMS = ["det", "det", "semidet", "multi", "multi", "nondet",
                "cc_multi", "cc_nondet", None, None]


class Gen:
    def __init__(self, rng, preds, depth=3, scopes=False, io=False):
        self.rng = rng
        self.scopes = scopes
        self.io = io
        self.preds = preds
        self.depth = depth
        self.count = 0
        self.ints = []
        self.dead = []

    def fresh(self, prefix):
        self.count += 1
        return "%s%d" % (prefix, self.count)

    def number(self):
        if self.ints and self.rng.random() < 0.6:
            return self.rng.choice(self.ints)
        return str(self.rng.randint(1, 3))

    def value(self, cur):
        """The value to update: now and then one already clobbered."""
        if self.dead and self.rng.random() < 0.04:
            return self.rng.choice(self.dead)
        return cur

    def choice(self):
        """A goal that binds a new integer, of some determinism."""
        x = self.fresh("X")
        goal = self.rng.choice(["m(%s)" % x, "m(%s)" % x, "nd(%s, %s)" % (self.number(), x),
                                "cm(%s)" % x, "%s = %s" % (x, self.number())])
        self.ints.append(x)
        return goal

    def test(self):
        return self.rng.choice(["sd(%s)" % self.number(), "%s > 1" % self.number()])

    def step(self, cur, depth):
        """A goal from the value `cur`, and the value after it."""
        r = self.rng.random()
        nxt = self.fresh("T")
        if r < 0.25:
            self.dead.append(cur)
            return "updu(%s, %s, %s)" % (self.number(), self.value(cur), nxt), nxt
        if r < 0.33:
            self.dead.append(cur)
            return "upd(%s, %s, %s)" % (self.number(), self.value(cur), nxt), nxt
        if r < 0.38:
            if self.io:
                return "%s = %s" % (nxt, cur), nxt
            return "%s = t(%s)" % (nxt, self.number()), nxt
        if r < 0.42:
            return "%s = %s" % (nxt, cur), nxt
        if r < 0.45:
            return "peek(%s, %s)" % (cur, self.fresh("Y")), cur
        if r < 0.55:
            self.dead.append(cur)
            name, modes = self.rng.choice(self.preds)
            return "%s(%s, %s)" % (name, cur, nxt), nxt
        if r < 0.67:
            return self.choice(), cur
        if r < 0.75:
            return self.test(), cur
        if depth <= 0:
            return "true", cur
        if r < 0.84:
            branches = []
            for _ in range(self.rng.choice([2, 2, 3])):
                branches.append(self.branch(cur, nxt, depth - 1))
            return "( " + " ; ".join(branches) + " )", nxt
        if r < 0.92:
            ints = list(self.ints)
            cond, mid = self.sequence(cur, depth - 1, self.rng.randint(1, 2))
            then = self.finish(mid, nxt, depth - 1)
            self.ints = list(ints)
            other = self.branch(cur, nxt, depth - 1)
            self.ints = ints
            return "( if %s then %s else %s )" % (cond, then, other), nxt
        if self.scopes and r >= 0.92 and self.rng.random() < 0.7:
            return self.scoped(cur, nxt, depth), nxt
        if r < 0.96:
            ints = list(self.ints)
            inner, _ = self.sequence(cur, depth - 1, self.rng.randint(1, 2))
            self.ints = ints
            return "\\+ ( %s )" % inner, cur
        ints = list(self.ints)
        inner = self.branch(cur, nxt, depth - 1)
        self.ints = ints
        return "promise_equivalent_solutions [%s] ( %s )" % (nxt, inner), nxt

    def scoped(self, cur, nxt, depth):
        """A goal from `cur` to `nxt` of those --scopes adds."""
        r = self.rng.random()
        ints = list(self.ints)
        if r < 0.3:
            a, b, p = self.fresh("A"), self.fresh("B"), self.fresh("P")
            mi, mo = self.rng.choice([("di", "uo"), ("di", "uo"), ("mdi", "muo")])
            det = self.rng.choice(LAMBDA_DETERMINISMS)
            body = self.finish(a, b, depth - 1)
            self.ints = ints
            return "%s = (pred(%s::%s, %s::%s) is %s :- %s), %s(%s, %s)" % (
                p, a, mi, b, mo, det, body, p, cur, nxt)
        if r < 0.5:
            p = self.fresh("P")
            closure = self.rng.choice(["mu", "mmu", "updu(%s)" % self.number(),
                                       "upd(%s)" % self.number()])
            if not closure.startswith("m"):
                self.dead.append(cur)
            return "%s = %s, %s(%s, %s)" % (p, closure, p, cur, nxt)
        if r < 0.7:
            x = self.fresh("X")
            binding = self.rng.choice(["m(%s)" % x, "nd(%s, %s)" % (self.number(), x),
                                       "m(%s), %s > 1" % (x, x)])
            self.ints.append(x)
            rest = self.finish(cur, nxt, depth - 1)
            self.ints = ints
            return ("promise_equivalent_solution_sets [%s] ( arbitrary [%s] ( %s ), %s )"
                    % (nxt, x, binding, rest))
        if r < 0.9:
            det = self.rng.choice(LAMBDA_DETERMINISMS)
            inner = self.branch(cur, nxt, depth - 1)
            return "require_%s ( %s )" % (det, inner)
        x = self.fresh("X")
        test = "%s > %s" % (x, self.number())
        self.ints = ints
        return "( all [%s] ( %s => %s ) -> %s = %s ; %s = %s )" % (
            x, self.choice_of(x), test, nxt, cur, nxt, cur)

    def choice_of(self, x):
        return self.rng.choice(["m(%s)" % x, "nd(%s, %s)" % (self.number(), x)])

    def sequence(self, cur, depth, length):
        goals = []
        for _ in range(length):
            goal, cur = self.step(cur, depth)
            goals.append(goal)
        return ", ".join(goals), cur

    def finish(self, cur, out, depth):
        """Goals from `cur` that end with `out` bound."""
        goals = []
        for _ in range(self.rng.randint(0, 2)):
            goal, cur = self.step(cur, depth)
            goals.append(goal)
        goals.append("%s = %s" % (out, cur))
        return ", ".join(goals)

    def branch(self, cur, out, depth):
        ints = list(self.ints)
        text = self.finish(cur, out, depth)
        self.ints = ints
        return text


def predicate(rng, preds, n, scopes, io):
    name, (mi, mo) = preds[n]
    det = rng.choice(DETERMINISMS)
    gen = Gen(rng, preds, scopes=scopes, io=io)
    goals = []
    cur = "T0"
    for _ in range(rng.randint(2, 6)):
        goal, cur = gen.step(cur, gen.depth)
        goals.append(goal)
    goals.append("T = %s" % cur)
    decl = ":- pred %s(t::%s, t::%s)%s." % (name, mi, mo, " is " + det if det else "")
    clause = "%s(T0, T) :-\n    %s." % (name, ",\n    ".join(goals))
    return decl, clause, det is not None


def module(rng, name, scopes, io=False, ui=False):
    count = rng.randint(4, 8)
    preds = [("p%d" % i, rng.choice(MODES)) for i in range(count)]
    exported, local, clauses = [], [], []
    for n in range(count):
        decl, clause, declared = predicate(rng, preds, n, scopes, io)
        (exported if declared else local).append(decl)
        clauses.append(clause)
    helpers = IO_HELPERS if io else HELPERS
    if ui:
        helpers = re.sub(r":- pred peek\(t::in, int::out\) is det\.\npeek\(.*\)\.\n",
                         ":- pred peek(t::ui, int::out) is det.\npeek(_, 1).\n", helpers)
    scope_helpers = IO_SCOPE_HELPERS if io else SCOPE_HELPERS
    return ((IO_PRELUDE if io else PRELUDE).format(name=name) +
            "\n".join(exported) + "\n\n" +
            ":- implementation.\n:- import_module int.\n\n" + helpers +
            (scope_helpers if scopes else "") + "\n" +
            "\n".join(local) + "\n\n" + "\n\n".join(clauses) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--scopes", action="store_true")
    parser.add_argument("--io", action="store_true")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("dir")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random("%d-%d" % (args.seed, i))
        name = "u%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(module(rng, name, args.scopes, args.io, args.ui))


if __name__ == "__main__":
    main()
