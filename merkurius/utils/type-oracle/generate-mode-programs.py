#!/usr/bin/env python3
"""Generate random Mercury modules for comparing mode analysis with mmc's.

Each module has predicates with random argument types and modes, and
clauses whose goals are unifications and calls, generated in an order in
which each goal's inputs are bound and then shuffled, so that mode
analysis has to schedule them. Every predicate is declared `nondet`, so
that determinism analysis, which mmc runs with mode analysis, reports
nothing. A few clauses get a mutation that may make them mode-incorrect:
an output left unbound, or two free variables unified. compare-types.py
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
            if r < 0.4:
                self.call()
            elif r < 0.7:
                self.deconstruct()
            else:
                self.make_bound(rng.choice(TYPES))
        for var, mode in zip(head, modes):
            if mode == "out":
                value = self.make_bound(self.vars[var])
                self.goals.append("%s = %s" % (var, value))
        mutation = None
        if rng.random() < 0.25 and self.goals:
            kind = rng.choice(["drop", "free"])
            outputs = [g for g in self.goals if g.startswith("H")]
            if kind == "drop" and outputs:
                # Only an output's binding, which leaves no type ambiguous.
                self.goals.remove(rng.choice(outputs))
                mutation = "an output left unbound"
            else:
                x, y = self.fresh("int"), self.fresh("int")
                self.goals.append("%s = %s" % (x, y))
                self.goals.append("%s(%s)" % ("use", x))
                self.goals.append("%s(%s)" % ("use", y))
                mutation = "two free variables unified"
        rng.shuffle(self.goals)
        body = ",\n    ".join(self.goals) if self.goals else "true"
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
        # One clause, generated for the first mode: with more modes it is
        # usually wrong for the others, which mode analysis must find too.
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
