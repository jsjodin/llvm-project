#!/usr/bin/env python3
"""Generate random Mercury modules for comparing mode analysis with mmc's.

Each module has predicates with random argument types and modes, and
clauses whose goals are unifications, calls, disjunctions and
if-then-elses that bind the same variables in each branch, and negated
tests, generated in an order in which each goal's inputs are bound and
then shuffled, so that mode analysis has to schedule them. A predicate
has one to three clauses. Every predicate is declared `nondet`, so
that determinism analysis, which mmc runs with mode analysis, reports
nothing. A few clauses get a mutation that may make them mode-incorrect:
an output left unbound, two free variables unified, a variable bound in
one branch only, or one bound only in a negation. compare-types.py
--modes --error-lines then checks that merkc and mmc report mode errors
on the same lines. The constructors of t have names of two letters, since
a one-letter constant could also be a character.

A program is mode-correct by construction except for its mutations, so a
difference is a difference in mode analysis, not in the programs.
"""

import argparse
import os
import random

TYPES = ["int", "t", "list(int)"]
# The constructors of t, with their argument types.
CONSTRUCTORS = [("ta", []), ("tb", ["int"]), ("tc", ["int", "t"])]


class Compound:
    """A disjunction or an if-then-else: its branches' goals, with the
    condition's goal first for an if-then-else, and the variables each
    branch binds."""

    def __init__(self, kind, branches, targets, condition=None):
        self.kind = kind
        self.branches = branches
        self.targets = targets
        self.condition = condition

    def render(self):
        def body(goals):
            return ",\n        ".join(goals) if goals else "true"
        if self.kind == "disj":
            return "(\n        " + "\n    ;\n        ".join(body(b) for b in self.branches) + "\n    )"
        then, otherwise = self.branches
        return ("( if " + self.condition + " then\n        " + body(then) +
                "\n    else\n        " + body(otherwise) + "\n    )")


class Clause:
    def __init__(self, rng, preds, index):
        self.rng = rng
        self.preds = preds
        self.index = index
        self.vars = {}  # name -> type
        self.bound = set()
        self.goals = []
        self.count = 0

    def fresh(self, type_):
        self.count += 1
        name = "V%d" % self.count
        self.vars[name] = type_
        return name

    def bound_of(self, type_):
        return [v for v in sorted(self.bound) if self.vars[v] == type_]

    def make_bound(self, type_, depth=0):
        """A goal that binds a new variable of the type; its name."""
        options = self.bound_of(type_)
        rng = self.rng
        var = self.fresh(type_)
        choice = rng.random()
        if options and choice < 0.3:
            self.goals.append("%s = %s" % (var, rng.choice(options)))
        elif type_ == "int":
            if choice < 0.6 or depth > 2:
                self.goals.append("%s = %d" % (var, rng.randint(0, 9)))
            else:
                x = self.make_bound("int", depth + 1)
                y = self.make_bound("int", depth + 1)
                self.goals.append("%s = %s + %s" % (var, x, y))
        elif type_ == "list(int)":
            if choice < 0.6 or depth > 2:
                self.goals.append("%s = []" % var)
            else:
                h = self.make_bound("int", depth + 1)
                t = self.make_bound("list(int)", depth + 1)
                self.goals.append("%s = [%s | %s]" % (var, h, t))
        else:
            name, args = rng.choice(CONSTRUCTORS if depth < 2 else CONSTRUCTORS[:2])
            values = [self.make_bound(a, depth + 1) for a in args]
            self.goals.append("%s = %s" % (var, name + ("(%s)" % ", ".join(values) if values else "")))
        self.bound.add(var)
        return var

    def deconstruct(self):
        """A goal that takes apart a bound term, binding its parts."""
        rng = self.rng
        lists = self.bound_of("list(int)")
        terms = self.bound_of("t")
        if lists and rng.random() < 0.5:
            var = rng.choice(lists)
            h, t = self.fresh("int"), self.fresh("list(int)")
            self.goals.append("%s = [%s | %s]" % (var, h, t))
            self.bound.update([h, t])
        elif terms:
            var = rng.choice(terms)
            name, args = rng.choice(CONSTRUCTORS)
            parts = [self.fresh(a) for a in args]
            self.goals.append("%s = %s" % (var, name + ("(%s)" % ", ".join(parts) if parts else "")))
            self.bound.update(parts)

    def call(self):
        """A call of an earlier predicate, in one of its modes."""
        rng = self.rng
        callees = self.preds[:self.index]
        if not callees:
            return
        name, types, modes_list = rng.choice(callees)
        modes = rng.choice(modes_list)
        args = []
        outputs = []
        for type_, mode in zip(types, modes):
            if mode == "in":
                args.append(self.make_bound(type_))
            else:
                var = self.fresh(type_)
                args.append(var)
                outputs.append(var)
        self.goals.append("%s(%s)" % (name, ", ".join(args)))
        self.bound.update(outputs)

    def branch(self, targets):
        """Goals that bind each target, in a list of their own; what else
        they bind is local to them."""
        saved_goals, saved_bound = self.goals, set(self.bound)
        self.goals = []
        for var in targets:
            value = self.make_bound(self.vars[var])
            self.goals.append("%s = %s" % (var, value))
        self.rng.shuffle(self.goals)
        goals = self.goals
        self.goals, self.bound = saved_goals, saved_bound
        return goals

    def test(self):
        """A test of a bound variable, or None."""
        rng = self.rng
        options = sorted(self.bound)
        if not options:
            return None
        var = rng.choice(options)
        type_ = self.vars[var]
        if type_ == "int":
            return "%s = %d" % (var, rng.randint(0, 9))
        if type_ == "list(int)":
            return "%s = []" % var
        return "%s = %s" % (var, rng.choice(["ta", "tb(%d)" % rng.randint(0, 9)]))

    def compound(self):
        rng = self.rng
        kind = rng.choice(["disj", "ite", "not"])
        if kind == "not":
            test = self.test()
            if test:
                self.goals.append("not (%s)" % test)
            return
        targets = [self.fresh(rng.choice(TYPES)) for _ in range(rng.randint(1, 2))]
        if kind == "disj":
            branches = [self.branch(targets) for _ in range(rng.randint(2, 3))]
            self.goals.append(Compound(kind, branches, targets))
        else:
            condition = self.test() or "true"
            branches = [self.branch(targets), self.branch(targets)]
            self.goals.append(Compound(kind, branches, targets, condition))
        self.bound.update(targets)

    def generate(self, types, modes):
        rng = self.rng
        head = []
        for i, (type_, mode) in enumerate(zip(types, modes)):
            var = "H%d" % (i + 1)
            self.vars[var] = type_
            head.append(var)
            if mode == "in":
                self.bound.add(var)
        for _ in range(rng.randint(0, 4)):
            r = rng.random()
            if r < 0.3:
                self.call()
            elif r < 0.5:
                self.deconstruct()
            elif r < 0.75:
                self.compound()
            else:
                self.make_bound(rng.choice(TYPES))
        for var, mode in zip(head, modes):
            if mode == "out":
                value = self.make_bound(self.vars[var])
                self.goals.append("%s = %s" % (var, value))
        mutation = None
        if rng.random() < 0.3 and self.goals:
            kind = rng.choice(["drop", "free", "branch", "negation"])
            outputs = [g for g in self.goals if isinstance(g, str) and g.startswith("H")]
            compounds = [g for g in self.goals if isinstance(g, Compound)]
            if kind == "drop" and outputs:
                # Only an output's binding, which leaves no type ambiguous.
                self.goals.remove(rng.choice(outputs))
                mutation = "an output left unbound"
            elif kind == "branch" and compounds:
                compound = rng.choice(compounds)
                goals = rng.choice(compound.branches)
                target = rng.choice(compound.targets)
                goals[:] = [g for g in goals if not g.startswith(target + " = ")]
                mutation = "a variable bound in one branch only"
            elif kind == "negation":
                x = self.fresh("int")
                self.goals.append("not (%s = %d)" % (x, rng.randint(0, 9)))
                self.goals.append("use(%s)" % x)
                mutation = "a variable bound only in a negation"
            else:
                x, y = self.fresh("int"), self.fresh("int")
                self.goals.append("%s = %s" % (x, y))
                self.goals.append("%s(%s)" % ("use", x))
                self.goals.append("%s(%s)" % ("use", y))
                mutation = "two free variables unified"
        rng.shuffle(self.goals)
        goals = [g.render() if isinstance(g, Compound) else g for g in self.goals]
        body = ",\n    ".join(goals) if goals else "true"
        return head, body, mutation


def module(rng, name):
    lines = [
        "% Generated by utils/type-oracle/generate-mode-programs.py.",
        ":- module %s." % name,
        ":- interface.",
        ":- import_module list.",
        ":- type t ---> ta ; tb(int) ; tc(int, t).",
        ":- pred use(int::in) is det.",
        ":- implementation.",
        ":- import_module int.",
        "use(_).",
    ]
    preds = []
    for i in range(rng.randint(2, 6)):
        arity = rng.randint(1, 3)
        types = [rng.choice(TYPES) for _ in range(arity)]
        modes_list = []
        for _ in range(rng.choice([1, 1, 2])):
            modes = [rng.choice(["in", "out"]) for _ in range(arity)]
            if modes not in modes_list:
                modes_list.append(modes)
        preds.append(("p%d" % i, types, modes_list))
    for index, (pname, types, modes_list) in enumerate(preds):
        lines.append(":- pred %s(%s)." % (pname, ", ".join(types)))
        for modes in modes_list:
            lines.append(":- mode %s(%s) is nondet." % (pname, ", ".join(modes)))
        # Clauses generated for the first mode: with more modes they are
        # usually wrong for the others, which mode analysis must find too.
        for _ in range(rng.choice([1, 1, 2, 3])):
            clause = Clause(rng, preds, index)
            head, body, mutation = clause.generate(types, modes_list[0])
            if mutation:
                lines.append("% mutation: " + mutation)
            lines.append("%s(%s) :-\n    %s." % (pname, ", ".join(head), body))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("directory")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.directory, exist_ok=True)
    for i in range(args.count):
        name = "modes_%d_%d" % (args.seed, i)
        with open(os.path.join(args.directory, name + ".m"), "w") as f:
            f.write(module(rng, name))


if __name__ == "__main__":
    main()
