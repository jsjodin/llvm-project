#!/usr/bin/env python3
"""Generate random Mercury modules for comparing type checking and
inference with mmc's (compare-types.py).

Each module has predicates and functions with hidden signatures, random
types with type variables, and clauses built to fit them: unifications
with terms of the right types, calls of each other, so recursion and
mutual recursion, calls of library functions, closures, lambda
expressions, higher-order calls and if-then-else. Some predicates declare
their hidden signatures, and the others' types are inferred, which may be
more general. A module may have one mutation, an integer replaced by a
string, which makes it ill-typed, or not.

With --classes, each module also has type classes, show, loud <= show and
conv(A, B), with instances, and each hidden signature constrains some of
its type variables by show or loud, as the declared ones declare. Terms
call the methods on types that satisfy them, the predicate's constrained
type variables among them, and calls and closures bind a predicate's type
variables to types that satisfy its constraints. With --dependencies too,
there is a type class measure(A, B) <= (A -> B), whose method's result
only improvement by its instances determines, as in measure(1) ++ "".
With --existentials too, there are existentially typed functions, whose
results only their own existential constraints and polymorphic callees
accept, as in show(opaque(1)) or list.length([opaque(1)]), and rarely
two calls' types in one list, which is a type error. With
--existential-constructors too, there is an existentially typed
constructor, built with 'new shown'(...) and taken apart by a function,
and rarely built without new, which is a type error.

The modules are a function of the seed: python3 generate-random-programs.py
--seed 1 --count 20 DIR writes DIR/random_1_0.m ... DIR/random_1_19.m.
"""

import argparse
import os
import random

# Types: ("int",), ("string",), ("char",), ("float",), ("list", T),
# ("maybe", T), ("pair", A, B), ("tuple", (A, ...)), ("tree", T),
# ("colour",), ("func", (A, ...), R), ("pred", (A, ...)), ("var", name).

GROUND = [("int",), ("string",), ("char",), ("float",), ("colour",)]


# The instances: those of show and loud for each type, and conv's.
CLASS_LINES = [
    ":- typeclass show(T) where [ func show(T) = string ].",
    ":- typeclass loud(T) <= show(T) where [ func loud(T) = string ].",
    ":- typeclass conv(A, B) where [ func conv(A) = B ].",
    ":- instance show(int) where [ show(X) = string.int_to_string(X) ].",
    ":- instance show(string) where [ show(X) = X ].",
    ":- instance show(list(T)) <= show(T) where [ show(_) = \"[...]\" ].",
    ":- instance show(maybe(T)) <= show(T) where [ show(_) = \"maybe\" ].",
    ":- instance loud(int) where [ loud(X) = show(X) ++ \"!\" ].",
    ":- instance loud(list(T)) <= loud(T) where [ loud(_) = \"[...]!\" ].",
    ":- instance conv(int, string) where [ conv(X) = string.int_to_string(X) ].",
    ":- instance conv(string, int) where [ conv(X) = string.length(X) ].",
]
EXISTENTIAL_LINES = [
    ":- some [T] func opaque(int) = T => show(T).",
    "opaque(X) = X.",
    ":- some [T] func opaque_list(string) = list(T) => show(T).",
    "opaque_list(S) = [S, S].",
]
CONSTRUCTOR_LINES = [
    ":- type shown ---> some [T] (shown(T) => show(T)).",
    ":- func shown_text(shown) = string.",
    "shown_text(shown(X)) = show(X).",
    ":- func shown_length(shown) = int.",
    "shown_length(S) = string.length(shown_text(S)).",
]
DEPENDENCY_LINES = [
    ":- typeclass measure(A, B) <= (A -> B) where [ func measure(A) = B ].",
    ":- instance measure(int, string) where [ measure(X) = string.int_to_string(X) ].",
    ":- instance measure(list(T), list(U)) <= measure(T, U) where [ measure(_) = [] ].",
]
CLASS_ARGUMENTS = [("int",), ("string",), ("list", ("int",)), ("maybe", ("string",)),
                   ("list", ("list", ("int",))), ("maybe", ("list", ("int",)))]


class Module:
    def __init__(self, rng, name, mutate, classes=False, dependencies=False,
                 existentials=False, constructors=False):
        self.rng = rng
        self.name = name
        self.preds = []
        self.literals = 0
        self.mutate_at = None
        self.mutate = mutate
        self.classes = classes or dependencies or existentials or constructors
        self.dependencies = dependencies
        self.existentials = existentials or constructors
        self.constructors = constructors

    def satisfies(self, t, cls, needs):
        """Whether the instances, and `needs`, the classes that a
        predicate's type variables are constrained by, give cls(t)."""
        k = t[0]
        if k == "var":
            have = needs.get(t[1], set())
            return cls in have or (cls == "show" and "loud" in have)
        if k == "int":
            return True
        if k == "string":
            return cls == "show"
        if k == "list":
            return self.satisfies(t[1], cls, needs)
        if k == "maybe":
            return cls == "show" and self.satisfies(t[1], cls, needs)
        return False

    def binds_needs(self, pred, binding, needs):
        """Whether the binding of pred's type variables satisfies its
        constraints, in a clause whose type variables have `needs`."""
        return all(v not in binding or self.satisfies(binding[v], cls, needs)
                   for v, classes in pred["needs"].items() for cls in classes)

    # Types.

    def random_type(self, depth, vars_):
        rng = self.rng
        if vars_ and rng.random() < 0.25:
            return ("var", rng.choice(vars_))
        if depth <= 0 or rng.random() < 0.45:
            return rng.choice(GROUND)
        choice = rng.choice(["list", "list", "maybe", "pair", "tuple", "tree", "func", "pred"])
        d = depth - 1
        if choice in ("list", "maybe", "tree"):
            return (choice, self.random_type(d, vars_))
        if choice == "pair":
            return ("pair", self.random_type(d, vars_), self.random_type(d, vars_))
        if choice == "tuple":
            return ("tuple", tuple(self.random_type(d, vars_) for _ in range(rng.randint(2, 3))))
        if choice == "func":
            return ("func", tuple(self.random_type(d, vars_) for _ in range(rng.randint(1, 2))), self.random_type(d, vars_))
        return ("pred", tuple(self.random_type(d, vars_) for _ in range(rng.randint(1, 2))))

    def type_text(self, t):
        k = t[0]
        if k == "var":
            return t[1]
        if k == "char":
            return "character"
        if k in ("int", "string", "float", "colour"):
            return k
        if k in ("list", "maybe", "tree"):
            return "%s(%s)" % (k, self.type_text(t[1]))
        if k == "pair":
            return "pair(%s, %s)" % (self.type_text(t[1]), self.type_text(t[2]))
        if k == "tuple":
            return "{%s}" % ", ".join(self.type_text(a) for a in t[1])
        if k == "func":
            return "(func(%s) = %s)" % (", ".join(self.type_text(a) for a in t[1]), self.type_text(t[2]))
        return "pred(%s)" % ", ".join(self.type_text(a) for a in t[1])

    def substitute(self, t, binding):
        if t[0] == "var":
            return binding.get(t[1], t)
        if t[0] in ("list", "maybe", "tree"):
            return (t[0], self.substitute(t[1], binding))
        if t[0] == "pair":
            return ("pair", self.substitute(t[1], binding), self.substitute(t[2], binding))
        if t[0] == "tuple":
            return ("tuple", tuple(self.substitute(a, binding) for a in t[1]))
        if t[0] == "func":
            return ("func", tuple(self.substitute(a, binding) for a in t[1]), self.substitute(t[2], binding))
        if t[0] == "pred":
            return ("pred", tuple(self.substitute(a, binding) for a in t[1]))
        return t

    def match(self, pattern, t, binding):
        """Binds the pattern's variables so that it is t."""
        if pattern[0] == "var":
            if pattern[1] in binding:
                return binding[pattern[1]] == t
            binding[pattern[1]] = t
            return True
        if pattern[0] != t[0]:
            return False
        if pattern[0] in ("list", "maybe", "tree"):
            return self.match(pattern[1], t[1], binding)
        if pattern[0] == "pair":
            return self.match(pattern[1], t[1], binding) and self.match(pattern[2], t[2], binding)
        if pattern[0] in ("tuple", "pred"):
            return len(pattern[1]) == len(t[1]) and all(self.match(a, b, binding) for a, b in zip(pattern[1], t[1]))
        if pattern[0] == "func":
            return (
                len(pattern[1]) == len(t[1])
                and all(self.match(a, b, binding) for a, b in zip(pattern[1], t[1]))
                and self.match(pattern[2], t[2], binding)
            )
        return True

    def instantiate(self, pred, binding, env):
        """The predicate's signature with each variable not yet bound bound
        to a random type, one that satisfies its constraints."""
        for v in pred["vars"]:
            if v not in binding:
                choices = [t for t in GROUND + [("list", ("int",))]
                           if all(self.satisfies(t, cls, {}) for cls in pred["needs"].get(v, ()))]
                binding[v] = self.rng.choice(choices)
        return [self.substitute(a, binding) for a in pred["args"]]

    # Terms of a type, in a clause whose variables have the types in env.

    def fresh(self, clause, t):
        clause["count"] += 1
        name = "V%d" % clause["count"]
        clause["env"][name] = t
        return name

    def literal_int(self):
        self.literals += 1
        if self.mutate and self.mutate_at is None and self.rng.random() < 0.15:
            self.mutate_at = self.literals
            return '"oops"'
        return str(self.rng.randint(0, 9))

    def term(self, clause, t, depth):
        rng = self.rng
        env = clause["env"]
        same = [v for v, vt in env.items() if vt == t]
        if same and (depth <= 0 or rng.random() < 0.35):
            return rng.choice(same)
        k = t[0]
        if k == "var":
            return rng.choice(same) if same else "_"
        options = []
        if k == "int":
            options += [self.literal_int, lambda: "(%s + %s)" % (self.term(clause, t, depth - 1), self.term(clause, t, depth - 1))]
            options += [lambda: "list.length(%s)" % self.term(clause, ("list", rng.choice(GROUND)), depth - 1)]
            options += [lambda: "string.length(%s)" % self.term(clause, ("string",), depth - 1)]
            if self.classes:
                options += [lambda: "conv(%s)" % self.class_argument(clause, ("string",), depth - 1)]
            if self.constructors:
                options += [lambda: "shown_length('new shown'(%s))" % self.term(clause, ("string",), depth - 1)]
            if self.existentials:
                options += [lambda: "list.length([opaque(%s)])" % self.term(clause, t, depth - 1),
                            lambda: "list.length(opaque_list(%s))" % self.term(clause, ("string",), depth - 1)]
                if rng.random() < 0.1:
                    options += [lambda: "list.length([opaque(%s), opaque(%s)])" % (
                        self.term(clause, t, depth - 1), self.term(clause, t, depth - 1))]
        elif k == "string":
            options += [lambda: '"%s"' % rng.choice("abc"), lambda: "(%s ++ %s)" % (self.term(clause, t, depth - 1), self.term(clause, t, depth - 1))]
            if self.classes:
                for cls in ("show", "loud"):
                    arguments = [a for a in CLASS_ARGUMENTS if self.satisfies(a, cls, {})]
                    # The clause's constrained type variables, with a
                    # variable of the type to apply the method to.
                    arguments += [vt for vt in set(env.values())
                                  if vt[0] == "var" and self.satisfies(vt, cls, clause["needs"])]
                    options += [lambda cls=cls, arguments=arguments: "%s(%s)" % (
                        cls, self.class_argument(clause, rng.choice(arguments), depth - 1))]
                options += [lambda: "conv(%s)" % self.class_argument(clause, ("int",), depth - 1)]
            if self.dependencies:
                options += [lambda: "measure(%s)" % self.class_argument(clause, ("int",), depth - 1)]
            if self.constructors:
                arguments = [a for a in CLASS_ARGUMENTS if self.satisfies(a, "show", {})]
                options += [lambda: "shown_text('new shown'(%s))" % self.class_argument(
                    clause, rng.choice(arguments), depth - 1)]
                if rng.random() < 0.1:
                    options += [lambda: "shown_text(shown(%s))" % self.term(clause, ("int",), depth - 1)]
            if self.existentials:
                options += [lambda: "show(opaque(%s))" % self.term(clause, ("int",), depth - 1),
                            lambda: "show(list.det_head(opaque_list(%s)))" % self.term(clause, t, depth - 1)]
        elif k == "char":
            options += [lambda: "'%s'" % rng.choice("xyz")]
        elif k == "float":
            options += [lambda: "%d.5" % rng.randint(0, 9)]
        elif k == "colour":
            options += [lambda: rng.choice(["red", "green", "blue"])]
        elif k == "list":
            if self.dependencies and t[1] == ("string",):
                options += [lambda: "measure(%s)" % self.class_argument(clause, ("list", ("int",)), depth - 1)]
            options += [
                lambda: "[]",
                lambda: "[%s | %s]" % (self.term(clause, t[1], depth - 1), self.term(clause, t, depth - 1)),
                lambda: "[%s]" % ", ".join(self.term(clause, t[1], depth - 1) for _ in range(rng.randint(1, 2))),
                lambda: "list.reverse(%s)" % self.term(clause, t, depth - 1),
            ]
        elif k == "maybe":
            options += [lambda: "no", lambda: "yes(%s)" % self.term(clause, t[1], depth - 1)]
        elif k == "tree":
            options += [lambda: "leaf", lambda: "node(%s, %s, %s)" % (self.term(clause, t, depth - 2), self.term(clause, t[1], depth - 1), self.term(clause, t, depth - 2))]
        elif k == "pair":
            options += [lambda: "(%s - %s)" % (self.term(clause, t[1], depth - 1), self.term(clause, t[2], depth - 1))]
        elif k == "tuple":
            options += [lambda: "{%s}" % ", ".join(self.term(clause, a, depth - 1) for a in t[1])]
        elif k == "func":
            options += [lambda: self.lambda_func(clause, t, depth)]
        elif k == "pred":
            options += [lambda: self.lambda_pred(clause, t, depth)]
        # Calls of the module's functions whose results can have the type,
        # and closures of its predicates and functions.
        for pred in self.preds:
            binding = {}
            if pred["kind"] == "func" and self.match(pred["result"], t, binding) and \
                    self.binds_needs(pred, binding, clause["needs"]):
                options.append(lambda pred=pred, binding=binding: self.call_text(clause, pred, dict(binding), depth))
            if k in ("func", "pred") and pred["kind"] == k:
                arity = len(pred["args"]) - (1 if k == "func" else 0)
                binding = {}
                closure = ("func", tuple(pred["args"][:-1]), pred["args"][-1]) if k == "func" else ("pred", tuple(pred["args"]))
                if arity == len(t[1]) and self.match(closure, t, binding) and \
                        self.binds_needs(pred, binding, clause["needs"]):
                    options.append(lambda pred=pred: pred["name"])
        if depth <= 0:
            # A leaf, or terms of the type's parts, which are smaller.
            leaves = {
                "int": lambda: self.literal_int(),
                "string": lambda: '"%s"' % rng.choice("abc"),
                "list": lambda: "[]",
                "maybe": lambda: "no",
                "tree": lambda: "leaf",
            }
            if k in leaves:
                return leaves[k]()
            if k not in ("pair", "tuple", "func", "pred", "char", "float", "colour"):
                return rng.choice(options)()
            return options[0]()
        return rng.choice(options)()

    def class_argument(self, clause, t, depth):
        """A method's argument: a term of a ground type is qualified with
        it, since a term such as [] has more types, and the constraint would
        be on one that nothing determines."""
        if t[0] == "var":
            return self.term(clause, t, depth)
        return "(%s : %s)" % (self.term(clause, t, depth), self.type_text(t))

    def call_text(self, clause, pred, binding, depth):
        types = self.instantiate(pred, binding, clause["env"])
        arguments = [self.term(clause, a, depth - 1) for a in types[:-1]]
        return "%s(%s)" % (pred["name"], ", ".join(arguments)) if arguments else pred["name"]

    # A lambda expression's head variables, and the variables of its body
    # that do not occur outside it, are its own.

    def lambda_func(self, clause, t, depth):
        # The result is a variable of its own: `func(X) = Y`, with Y a
        # variable, would make Y a head variable of the lambda expression.
        outer = dict(clause["env"])
        names = [self.fresh(clause, a) for a in t[1]]
        body = self.term(clause, t[2], depth - 1)
        result = self.fresh(clause, t[2])
        clause["env"] = outer
        return "(func(%s) = %s :- %s = %s)" % (", ".join(names), result, result, body)

    def lambda_pred(self, clause, t, depth):
        outer = dict(clause["env"])
        names = [self.fresh(clause, a) for a in t[1]]
        goal = self.goal(clause, depth - 1)
        clause["env"] = outer
        return "(pred(%s) is semidet :- %s)" % (", ".join(n + "::in" for n in names), goal)

    # Goals.

    def goal(self, clause, depth):
        rng = self.rng
        env = clause["env"]
        options = ["unify", "unify", "call"]
        if any(t[0] in ("func", "pred") for t in env.values()):
            options += ["ho", "ho"]
        if depth > 0:
            options += ["ite"]
        choice = rng.choice(options)
        if choice == "unify":
            if env and rng.random() < 0.7:
                var = rng.choice(sorted(env))
                t = env[var]
            else:
                t = self.random_type(2, [])
                var = self.fresh(clause, t)
            return "%s = %s" % (var, self.term(clause, t, depth))
        if choice == "call":
            preds = [p for p in self.preds if p["kind"] == "pred"]
            if not preds:
                return "true"
            pred = rng.choice(preds)
            types = self.instantiate(pred, {}, env)
            arguments = [self.term(clause, a, depth - 1) for a in types]
            return "%s(%s)" % (pred["name"], ", ".join(arguments)) if arguments else pred["name"]
        if choice == "ho":
            var = rng.choice(sorted(v for v, t in env.items() if t[0] in ("func", "pred")))
            t = env[var]
            arguments = [self.term(clause, a, depth - 1) for a in t[1]]
            if t[0] == "pred":
                return "call(%s)" % ", ".join([var] + arguments) if rng.random() < 0.5 else "%s(%s)" % (var, ", ".join(arguments))
            return "%s = %s(%s)" % (self.term(clause, t[2], 0), var, ", ".join(arguments))
        condition = "%s > %s" % (self.term(clause, ("int",), 1), self.term(clause, ("int",), 1))
        return "( if %s then %s else %s )" % (condition, self.goal(clause, depth - 1), self.goal(clause, depth - 1))

    # Predicates and clauses.

    def make_preds(self):
        rng = self.rng
        for i in range(rng.randint(3, 8)):
            vars_ = ["T%d" % j for j in range(rng.randint(0, 2))]
            kind = rng.choice(["pred", "func"])
            arity = rng.randint(0, 3) + (1 if kind == "func" else 0)
            if kind == "pred" and arity == 0:
                arity = 1
            args = [self.random_type(2, vars_) for _ in range(arity)]
            used = set()

            def collect(t):
                if t[0] == "var":
                    used.add(t[1])
                for part in t[1:]:
                    if isinstance(part, tuple) and part and isinstance(part[0], str):
                        collect(part)
                    elif isinstance(part, tuple):
                        for p in part:
                            collect(p)

            for a in args:
                collect(a)
            needs = {}
            for v in sorted(used):
                roll = rng.random() if self.classes else 1
                if roll < 0.3:
                    needs[v] = {"show"}
                elif roll < 0.45:
                    needs[v] = {"loud"}
            self.preds.append({
                "needs": needs,
                "name": "%s%d" % (kind[0], i),
                "kind": kind,
                "vars": sorted(used),
                "args": args,
                "result": args[-1] if kind == "func" else None,
                "declared": rng.random() < 0.3,
            })

    def clause_text(self, pred):
        rng = self.rng
        clause = {"env": {}, "count": 0, "needs": pred["needs"]}
        heads = []
        for j, t in enumerate(pred["args"]):
            name = "H%d" % (j + 1)
            clause["env"][name] = t
            heads.append(name)
        goals = [self.goal(clause, 2) for _ in range(rng.randint(0, 3))]
        if pred["kind"] == "func":
            arguments = heads[:-1]
            result = heads[-1]
            head = "%s(%s) = %s" % (pred["name"], ", ".join(arguments), result) if arguments else "%s = %s" % (pred["name"], result)
            goals.append("%s = %s" % (result, self.term(clause, pred["args"][-1], 2)))
        else:
            head = "%s(%s)" % (pred["name"], ", ".join(heads))
        return head + (" :- " + ", ".join(goals) if goals else "") + "."

    def text(self):
        self.make_preds()
        lines = [
            "% Generated by utils/type-oracle/generate-random-programs.py.",
            ":- module %s." % self.name,
            ":- interface.",
            ":- pred main(int::out) is det.",
            ":- implementation.",
            ":- import_module float, int, list, maybe, pair, string.",
            ":- type colour ---> red ; green ; blue.",
            ":- type tree(T) ---> leaf ; node(tree(T), T, tree(T)).",
        ]
        if self.classes:
            lines += CLASS_LINES
        if self.dependencies:
            lines += DEPENDENCY_LINES
        if self.existentials:
            lines += EXISTENTIAL_LINES
        if self.constructors:
            lines += CONSTRUCTOR_LINES
        lines.append("main(0).")
        for pred in self.preds:
            if pred["declared"]:
                args = [self.type_text(a) for a in pred["args"]]
                constraints = ["%s(%s)" % (cls, v) for v, classes in sorted(pred["needs"].items())
                               for cls in sorted(classes)]
                where = " <= (%s)" % ", ".join(constraints) if constraints else ""
                if pred["kind"] == "func":
                    lines.append(":- func %s%s = %s%s." % (pred["name"], "(%s)" % ", ".join(args[:-1]) if args[:-1] else "", args[-1], where))
                else:
                    lines.append(":- pred %s(%s)%s." % (pred["name"], ", ".join(args), where))
            for _ in range(self.rng.randint(1, 2)):
                lines.append(self.clause_text(pred))
        return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--classes", action="store_true", help="with type classes")
    parser.add_argument("--existential-constructors", action="store_true",
                        help="with --existentials and an existentially typed constructor")
    parser.add_argument("--existentials", action="store_true",
                        help="with type classes and existentially typed functions")
    parser.add_argument("--dependencies", action="store_true",
                        help="with type classes and a functional dependency")
    parser.add_argument("directory")
    args = parser.parse_args()
    os.makedirs(args.directory, exist_ok=True)
    for i in range(args.count):
        rng = random.Random("%d/%d" % (args.seed, i))
        name = "random_%d_%d" % (args.seed, i)
        module = Module(rng, name, mutate=rng.random() < 0.25, classes=args.classes,
                        dependencies=args.dependencies,
                        existentials=args.existentials,
                        constructors=args.existential_constructors)
        with open(os.path.join(args.directory, name + ".m"), "w") as f:
            f.write(module.text())


if __name__ == "__main__":
    main()
