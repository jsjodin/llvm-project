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

Usage: generate-polymorphic-programs.py [--seed N] [--count N] DIR
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
    def __init__(self, name, func, inputs, output, own):
        self.name, self.func, self.inputs, self.output, self.own = \
            name, func, inputs, output, own


LIBRARY = [
    Callee("list.length", False, [("list", ("var", "A"))], ("int",), ["A"]),
    Callee("list.reverse", False, [("list", ("var", "A"))], ("list", ("var", "A")), ["A"]),
    Callee("list.append", False, [("list", ("var", "A")), ("list", ("var", "A"))],
           ("list", ("var", "A")), ["A"]),
    Callee("pair.fst", False, [("pair", ("var", "A"), ("var", "B"))], ("var", "A"), ["A", "B"]),
]


class Body:
    def __init__(self, rng, callees, own, env):
        self.rng, self.callees, self.own = rng, callees, own
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
        if kind == "list":
            if depth <= 0 or rng.random() < 0.4:
                return "[]"
            head = self.value(ty[1], depth - 1)
            return "[%s]" % head if head else "[]"
        if kind == "maybe":
            inner = self.value(ty[1], depth - 1) if depth > 0 and rng.random() < 0.6 else None
            return "yes(%s)" % inner if inner else "no"
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
            for var in callee.own:
                if var not in binding and not (unused and var in callee.void):
                    binding[var] = self.instance(depth)
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

    def goal(self, callee, arguments, out):
        if callee.func:
            self.goals.append("%s = %s(%s)" % (out, callee.name, ", ".join(arguments)))
        else:
            self.goals.append("%s(%s)" % (callee.name, ", ".join(arguments + [out])))


def module(rng, name):
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
        preds.append(callee)
    for callee in LIBRARY:
        callee.void = []
    callees = preds + LIBRARY
    lines = [":- module %s." % name, ":- interface.",
             # Not int, whose `-` would make X - Y ambiguous.
             ":- import_module list, pair, maybe.", ":- implementation."]
    for callee in preds:
        args = ", ".join(text(t) + "::in" for t in callee.inputs)
        if callee.func:
            lines.append(":- func %s(%s) = (%s::out) is det." %
                         (callee.name, args, text(callee.output)))
        else:
            lines.append(":- pred %s(%s, %s::out) is det." %
                         (callee.name, args, text(callee.output)))
        lines.extend(clauses(rng, callee, callees))
    return "\n".join(lines) + "\n"


def clauses(rng, callee, callees):
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
            out.append(clause(rng, callee, callees, own, heads, env))
        return out
    env = list(zip(names, callee.inputs))
    return [clause(rng, callee, callees, own, names, env)]


def clause(rng, callee, callees, own, heads, env):
    body = Body(rng, callees, own, env)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("dir")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    for i in range(args.count):
        rng = random.Random(args.seed * 100003 + i)
        name = "polymorphism_%d_%d" % (args.seed, i)
        with open(os.path.join(args.dir, name + ".m"), "w") as f:
            f.write(module(rng, name))


if __name__ == "__main__":
    main()
