#!/usr/bin/env python3
"""Generate random modules for compare-polymorphism.py.

Each module has polymorphic predicates and functions, with type variables
T and U, whose arguments are of types built from them, int, string,
lists, pairs, tuples and maybe; a few have an output type variable that
no input has, inside a list or a maybe. Their bodies build their outputs
from their inputs by constructions and calls of each other, recursive or
not, at types that bind the callees' type variables to the callers' own,
to types with no variables, or to types built from both, and of library
predicates (list.length, list.reverse, list.append, pair.fst); some calls only make values that nothing uses, now and then
of a callee whose output type variable nothing then determines, and so is
void. Some bodies test two values of a type variable, or compare them,
in an if-then-else, and some predicates have a clause for each of a list
input's functors. The modules are mode- and determinism-correct, so that
mmc checks them to the end.

With --classes, a module also has type classes, some with superclasses,
and now and then one with two parameters, each with a method; instances of
them for int, string, lists, maybes and pairs, those of the type
constructors with constraints on their type variables or not, whose
methods call the method, or a superclass's, on their parts; and class
constraints on the predicates' and functions' type variables. Bodies then
also call methods, and the predicates and functions with constraints, at
types whose constraints the caller's constraints or the instances satisfy.

Usage: generate-polymorphic-programs.py [--classes] [--seed N] [--count N] DIR
"""

import argparse
import os
import random

VARS = ["T", "U"]


def text(ty):
    kind = ty[0]
    if kind == "var":
        return ty[1]
    if kind in ("int", "string"):
        return kind
    if kind == "list":
        return "list(%s)" % text(ty[1])
    if kind == "maybe":
        return "maybe(%s)" % text(ty[1])
    if kind == "pair":
        return "pair(%s, %s)" % (text(ty[1]), text(ty[2]))
    return "{%s, %s}" % (text(ty[1]), text(ty[2]))


def variables(ty, out=None):
    out = [] if out is None else out
    if ty[0] == "var":
        if ty[1] not in out:
            out.append(ty[1])
    else:
        for part in ty[1:]:
            variables(part, out)
    return out


def substitute(ty, binding):
    if ty[0] == "var":
        return binding[ty[1]]
    if ty[0] in ("int", "string"):
        return ty
    return (ty[0],) + tuple(substitute(p, binding) for p in ty[1:])


def match(pattern, ty, binding):
    """Binds the variables of `pattern` so that it is `ty`, if it can be."""
    if pattern[0] == "var":
        if pattern[1] in binding:
            return binding[pattern[1]] == ty
        binding[pattern[1]] = ty
        return True
    if pattern[0] != ty[0] or len(pattern) != len(ty):
        return False
    return all(match(p, t, binding) for p, t in zip(pattern[1:], ty[1:]))


def random_type(rng, own, depth):
    """A type made of the variables `own` and the others."""
    choices = ["int", "string"] + ["var"] * (2 * len(own))
    if depth > 0:
        choices += ["list", "list", "pair", "tuple", "maybe"]
    kind = rng.choice(choices)
    if kind == "var":
        return ("var", rng.choice(own))
    if kind in ("int", "string"):
        return (kind,)
    if kind in ("list", "maybe"):
        return (kind, random_type(rng, own, depth - 1))
    return (kind, random_type(rng, own, depth - 1), random_type(rng, own, depth - 1))


class Callee:
    def __init__(self, name, func, inputs, output, own, constraints=()):
        self.name, self.func, self.inputs, self.output, self.own = \
            name, func, inputs, output, own
        # (class, [type]) on its own type variables.
        self.constraints = list(constraints)


class Classes:
    """A module's type classes and instances."""

    def __init__(self):
        # By name: its arity, superclasses and instances, each as its
        # types, patterns over its own type variables, and its
        # constraints, (class, [type variable]).
        self.arity = {}
        self.supers = {}
        self.instances = {}

    def closure(self, assumed):
        out = list(assumed)
        for cls, args in out:
            for sup in self.supers.get(cls, []):
                if (sup, args) not in out:
                    out.append((sup, args))
        return out

    def entails(self, cls, args, assumed, depth=0):
        if (cls, list(args)) in [(c, list(a)) for c, a in self.closure(assumed)]:
            return True
        if depth > 4:
            return False
        for types, constraints in self.instances.get(cls, []):
            binding = {}
            if all(match(t, a, binding) for t, a in zip(types, args)):
                return all(self.entails(c, [binding[v] for v in vs], assumed, depth + 1)
                           for c, vs in constraints)
        return False


LIBRARY = [
    Callee("list.length", False, [("list", ("var", "A"))], ("int",), ["A"]),
    Callee("list.reverse", False, [("list", ("var", "A"))], ("list", ("var", "A")), ["A"]),
    Callee("list.append", False, [("list", ("var", "A")), ("list", ("var", "A"))],
           ("list", ("var", "A")), ["A"]),
    Callee("pair.fst", False, [("pair", ("var", "A"), ("var", "B"))], ("var", "A"), ["A", "B"]),
]


class Body:
    def __init__(self, rng, callees, own, env, classes=None, assumed=()):
        self.rng, self.callees, self.own = rng, callees, own
        self.classes, self.assumed = classes, list(assumed)
        self.env = list(env)
        self.goals = []
        self.count = 0

    def fresh(self):
        self.count += 1
        return "V%d" % self.count

    def instance(self, depth):
        """A type to bind a callee's type variable to."""
        if self.own:
            return random_type(self.rng, self.own, min(depth, 1))
        return self.rng.choice([("int",), ("string",), ("list", ("int",))])

    def value(self, ty, depth):
        """A variable or term of type `ty`, with the goals that make it."""
        rng = self.rng
        known = [n for n, t in self.env if t == ty]
        if known and (ty[0] == "var" or rng.random() < 0.5):
            return rng.choice(known)
        if depth > 0 and rng.random() < 0.45:
            made = self.call(ty, depth - 1)
            if made:
                return made
        kind = ty[0]
        if kind == "var":
            if known:
                return rng.choice(known)
            return self.call(ty, depth - 1) if depth > 0 else None
        if kind == "int":
            return str(rng.randint(0, 9))
        if kind == "string":
            return '"%s"' % rng.choice("abc")
        # With type classes, a constant's type is given, so that a constraint
        # on it is determined.
        empty = "([] : %s)" % text(ty) if self.classes else "[]"
        if kind == "list":
            if depth <= 0 or rng.random() < 0.4:
                return empty
            head = self.value(ty[1], depth - 1)
            return "[%s]" % head if head else empty
        if kind == "maybe":
            inner = self.value(ty[1], depth - 1) if depth > 0 and rng.random() < 0.6 else None
            return "yes(%s)" % inner if inner else "(no : %s)" % text(ty) if self.classes else "no"
        first = self.value(ty[1], depth - 1)
        second = self.value(ty[2], depth - 1)
        if first is None or second is None:
            return None
        return ("(%s - %s)" if kind == "pair" else "{%s, %s}") % (first, second)

    def call(self, ty, depth, unused=False):
        """A call of a callee that makes a value of type `ty`, if one can,
        into a fresh variable; with `unused`, of any type, not used."""
        rng = self.rng
        callees = list(self.callees)
        rng.shuffle(callees)
        for callee in callees:
            binding = {}
            if not unused and not match(callee.output, ty, binding):
                continue
            if self.classes and callee.void and not unused:
                # An output type variable that no input has is determined
                # only where the output is used, which may not determine it.
                continue
            fixed = dict(binding)
            for _ in range(8 if callee.constraints else 1):
                binding = dict(fixed)
                for var in callee.own:
                    if var not in binding and not (unused and var in callee.void):
                        binding[var] = self.instance(depth)
                if self.satisfied(callee, binding):
                    break
            else:
                continue
            if unused and any(v not in binding for v in callee.own):
                # The output's type variable is left to void.
                arguments = []
                ok = True
                for t in callee.inputs:
                    a = self.value(substitute(t, binding), depth)
                    if a is None:
                        ok = False
                        break
                    arguments.append(a)
                if not ok:
                    continue
                self.goal(callee, arguments, "_")
                return "_"
            arguments = []
            for t in callee.inputs:
                a = self.value(substitute(t, binding), depth)
                if a is None:
                    break
                arguments.append(a)
            else:
                out = self.fresh()
                self.goal(callee, arguments, out)
                self.env.append((out, substitute(callee.output, binding)))
                return out
        return None

    def satisfied(self, callee, binding):
        """Whether the caller satisfies the callee's constraints there."""
        for cls, args in callee.constraints:
            if any(v not in binding for v in variables_of(args)):
                return False
            if not self.classes.entails(cls, [substitute(a, binding) for a in args],
                                        self.assumed):
                return False
        return True

    def goal(self, callee, arguments, out):
        if callee.func:
            self.goals.append("%s = %s(%s)" % (out, callee.name, ", ".join(arguments)))
        else:
            self.goals.append("%s(%s)" % (callee.name, ", ".join(arguments + [out])))


def variables_of(types):
    out = []
    for t in types:
        variables(t, out)
    return out


def constraint_text(cls, args):
    return "%s(%s)" % (cls, ", ".join(text(a) for a in args))


INSTANCE_TYPES = {
    "int": ([("int",)], []), "string": ([("string",)], []),
    "list": ([("list", ("var", "T"))], ["T"]), "maybe": ([("maybe", ("var", "T"))], ["T"]),
    "pair": ([("pair", ("var", "A"), ("var", "B"))], ["A", "B"]),
}


def make_classes(rng):
    """The type classes, their instances and their methods' callees, and
    the declarations."""
    classes = Classes()
    lines, methods = [], []
    count = rng.randint(1, 3)
    kinds = {}
    for k in range(count):
        cls = "c%d" % k
        supers = [s for s in ("c%d" % j for j in range(k)) if rng.random() < 0.5]
        classes.arity[cls] = 1
        classes.supers[cls] = []
        # Its superclasses', transitively.
        for sup in supers:
            for s in [sup] + classes.supers[sup]:
                if s not in classes.supers[cls]:
                    classes.supers[cls].append(s)
        allowed = list(INSTANCE_TYPES)
        for sup in supers:
            allowed = [t for t in allowed if t in kinds[sup]]
        kinds[cls] = [t for t in allowed if rng.random() < 0.7] or allowed[:1]
        lines.append(":- typeclass %s(T)%s where [ pred m%d(T::in, int::out) is det ]." %
                     (cls, " <= (%s)" % ", ".join("%s(T)" % s for s in supers) if supers else "",
                      k))
        methods.append(Callee("m%d" % k, False, [("var", "A")], ("int",), ["A"],
                              [(cls, [("var", "A")])]))
        classes.instances[cls] = []
        for kind in kinds[cls]:
            types, tvars = INSTANCE_TYPES[kind]
            # The constraints a superclass's instance has must follow.
            needed = set()
            for sup in classes.supers[cls]:
                for itypes, iconstraints in classes.instances[sup]:
                    if itypes == types:
                        needed |= {v for _, vs in iconstraints for v in vs}
            constrained = [v for v in tvars if v in needed or rng.random() < 0.6]
            constraints = [(cls, [v]) for v in constrained]
            classes.instances[cls].append((types, constraints))
            lines.append(":- instance %s(%s)%s where [ %s ]." % (
                cls, text(types[0]),
                " <= (%s)" % ", ".join("%s(%s)" % (c, v) for c, (v,) in constraints)
                if constraints else "",
                method_clause(rng, k, kind, constrained, classes.supers[cls])))
    if "c0" in classes.arity and rng.random() < 0.4:
        # A type class with two parameters.
        classes.arity["d"] = 2
        classes.supers["d"] = []
        lines.append(":- typeclass d(A, B) where [ pred dm(A::in, B::in, int::out) is det ].")
        methods.append(Callee("dm", False, [("var", "A"), ("var", "B")], ("int",), ["A", "B"],
                              [("d", [("var", "A"), ("var", "B")])]))
        classes.instances["d"] = [([("int",), ("string",)], [])]
        lines.append(":- instance d(int, string) where [ (dm(X, _, X)) ].")
        if "maybe" in kinds["c0"]:
            classes.instances["d"].append(
                ([("list", ("var", "A")), ("maybe", ("var", "B"))], [("c0", ["B"])]))
            lines.append(":- instance d(list(A), maybe(B)) <= c0(B) where [ (dm(_, M, N) :- "
                         "( if M = yes(X) then m0(X, N) else N = 0 )) ].")
    return classes, lines, methods


def method_clause(rng, k, kind, constrained, supers):
    """An instance's clause for method m<k>, on its type <kind>."""
    call = "m%d" % k
    if supers and rng.random() < 0.4:
        # A superclass's method on the same value, or on a part.
        call = "m%s" % rng.choice(supers)[1:]
    if kind == "int":
        return "(m%d(X, N) :- %s)" % (k, "N = X" if call == "m%d" % k else "%s(X, N)" % call)
    if kind == "string":
        return "(m%d(_, 2))" % k if call == "m%d" % k else "(m%d(X, N) :- %s(X, N))" % (k, call)
    if not constrained:
        return "(m%d(_, 0))" % k
    if kind == "list":
        return "(m%d(L, N) :- ( if L = [H | _] then %s(H, N) else N = 0 ))" % (k, call)
    if kind == "maybe":
        return "(m%d(M, N) :- ( if M = yes(X) then %s(X, N) else N = 1 ))" % (k, call)
    part = "A" if "A" in constrained else "B"
    return "(m%d(A - B, N) :- %s(%s, N), _ = %s)" % (
        k, call, part, "B" if part == "A" else "A")


def module(rng, name, with_classes=False):
    classes, class_lines, methods = (make_classes(rng) if with_classes
                                     else (None, [], []))
    preds = []
    for i in range(rng.randint(3, 6)):
        own = VARS[:rng.randint(0, 2)]
        func = rng.random() < 0.3
        inputs = [random_type(rng, own, 1) for _ in range(rng.randint(1, 3))]
        # An input of each type variable itself, since a body takes no term
        # apart, so that it can always make a value of each.
        inputs += [("var", v) for v in own if ("var", v) not in inputs]
        void = []
        if own and rng.random() < 0.15:
            # An output type variable that no input has.
            void = ["W"]
            output = (rng.choice(["list", "maybe"]), ("var", "W"))
        else:
            output = random_type(rng, own, 1)
        callee = Callee(("f%d" if func else "p%d") % i, func, inputs, output, own + void)
        callee.void = void
        if classes:
            for v in own:
                if rng.random() < 0.45:
                    cls = rng.choice([c for c in classes.arity if classes.arity[c] == 1])
                    callee.constraints.append((cls, [("var", v)]))
            if "d" in classes.arity and len(own) == 2 and rng.random() < 0.2:
                callee.constraints.append(("d", [("var", "T"), ("var", "U")]))
        preds.append(callee)
    for callee in LIBRARY + methods:
        callee.void = []
    callees = preds + LIBRARY + methods
    lines = [":- module %s." % name, ":- interface.",
             # Not int, whose `-` would make X - Y ambiguous.
             ":- import_module list, pair, maybe.", ":- implementation."]
    lines.extend(class_lines)
    for callee in preds:
        args = ", ".join(text(t) + "::in" for t in callee.inputs)
        where = (" <= (%s)" % ", ".join(constraint_text(c, a) for c, a in callee.constraints)
                 if callee.constraints else "")
        if callee.func:
            lines.append(":- func %s(%s) = (%s::out) is det%s." %
                         (callee.name, args, text(callee.output), where))
        else:
            lines.append(":- pred %s(%s, %s::out) is det%s." %
                         (callee.name, args, text(callee.output), where))
        lines.extend(clauses(rng, callee, callees, classes))
    return "\n".join(lines) + "\n"


def clauses(rng, callee, callees, classes=None):
    names = ["A%d" % (i + 1) for i in range(len(callee.inputs))]
    own = [v for v in callee.own if v not in callee.void]
    lists = [i for i, t in enumerate(callee.inputs) if t[0] == "list"]
    if lists and rng.random() < 0.3:
        # A clause for each of a list input's functors.
        i = rng.choice(lists)
        element = callee.inputs[i][1]
        out = []
        for pattern, env in (("[]", []), ("[H | Tl]", [("H", element), ("Tl", callee.inputs[i])])):
            heads = list(names)
            heads[i] = pattern
            env = env + [(n, t) for j, (n, t) in enumerate(zip(names, callee.inputs)) if j != i]
            out.append(clause(rng, callee, callees, own, heads, env, classes))
        return out
    env = list(zip(names, callee.inputs))
    return [clause(rng, callee, callees, own, names, env, classes)]


def clause(rng, callee, callees, own, heads, env, classes=None):
    body = Body(rng, callees, own, env, classes, callee.constraints)
    for _ in range(rng.randint(0, 2)):
        body.call(None, 1, unused=True)
    ty = substitute(callee.output, {v: ("var", v) for v in callee.own})
    result = None
    for _ in range(5):
        result = body.value(ty, 2)
        if result is not None:
            break
    if result is None:
        result = "[]" if ty[0] == "list" else "no"
    tested = [n for n, t in body.env if t[0] == "var"]
    if tested and rng.random() < 0.35:
        a, b = rng.choice(tested), rng.choice(tested)
        ta = dict(body.env)[a]
        same = [n for n, t in body.env if t == ta]
        b = rng.choice(same)
        other = body.value(ty, 1) or result
        if rng.random() < 0.5:
            body.goals.append("( if %s = %s then O = %s else O = %s )" % (a, b, result, other))
        else:
            body.goals.append("compare(R, %s, %s)" % (a, b))
            body.goals.append("( if R = (<) then O = %s else O = %s )" % (result, other))
    else:
        body.goals.append("O = %s" % result)
    if callee.func:
        head = "%s(%s) = O" % (callee.name, ", ".join(heads))
    else:
        head = "%s(%s)" % (callee.name, ", ".join(heads + ["O"]))
    return "%s :-\n    %s." % (head, ",\n    ".join(body.goals))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", action="store_true",
                        help="add type classes, instances and constraints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("dir")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(args.seed * 100003 + i)
        name = "polymorphism_%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(module(rng, name, args.classes))


if __name__ == "__main__":
    main()
