#!/usr/bin/env python3
"""Make the sound array stub from the manual's (array.m).

The library reference manual's array module works around mmc's missing
unique-input modes: its unique array insts are `ground`, so mmc checks no
array's uniqueness, and the predicates that read an array take it `in`.
The sound stub has the insts that the manual's own comments give, so that
arrays are unique, and gives each mode that takes arrays `in` copies
before it that take them `array_ui`, each mix of them, so that reading a
unique array keeps it unique. A declaration that gives types and modes together is split into a
`:- pred` or `:- func` declaration and its `:- mode` declarations. merkc
loads the sound stub unless -fmmc-uniqueness is given (design document,
"Uniqueness checking").

Usage: make-sound-array.py ARRAY_M OUTPUT
"""

import itertools
import re
import sys

INSTS = {
    ":- inst uniq_array(I) == array(I).": ":- inst uniq_array(I) == unique.",
    ":- inst uniq_array == uniq_array(ground).":
        ":- inst uniq_array == uniq_array(unique).",
    ":- inst mostly_uniq_array(I) == array(I).":
        ":- inst mostly_uniq_array(I) == mostly_unique.",
    ":- inst mostly_uniq_array == mostly_uniq_array(ground).":
        ":- inst mostly_uniq_array == mostly_uniq_array(mostly_unique).",
}

HEADER = """\
% The sound stub of the standard library module array, made by
% make-sound-array.py from array.m: arrays are unique, and each mode that
% takes an array `in` has copies before it that take it `array_ui`. Do not
% edit; edit array.m or the script.
"""


def split_top(text, sep=","):
    """`text` split at the `sep`s outside brackets."""
    parts, depth, cur = [], 0, ""
    for c in text:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts


def head(text):
    """The name, the arguments and the rest of `name(args) rest`."""
    m = re.match(r"('[^']*'|[a-z_][A-Za-z0-9_]*)\s*", text)
    name, rest = m.group(1), text[m.end():]
    if not rest.startswith("("):
        return name, [], rest
    depth = 0
    for i, c in enumerate(rest):
        depth += c == "("
        depth -= c == ")"
        if depth == 0:
            return name, split_top(rest[1:i]), rest[i + 1:].strip()
    raise ValueError(text)


def parse(decl):
    """A pred, func or mode declaration's parts: kind, name, arguments,
    the function result (or None) and the determinism text (or "")."""
    m = re.match(r":- (pred|func|mode) (.*)\.$", decl, re.S)
    kind, body = m.group(1), " ".join(m.group(2).split())
    det = ""
    d = re.search(r"\s+is\s+([a-z_]+)$", body)
    if d:
        det, body = d.group(1), body[:d.start()]
    result = None
    parts = split_top(body, "=")
    if len(parts) == 2:
        body, result = parts
        if result.startswith("(") and result.endswith(")"):
            result = result[1:-1].strip()
    name, arguments, _ = head(body.strip())
    return kind, name, arguments, result, det


def text(kind, name, arguments, result, det):
    out = ":- %s %s" % (kind, name)
    if arguments:
        out += "(" + ", ".join(arguments) + ")"
    if result is not None:
        out += " = " + (("(" + result + ")") if "::" in result else result)
    if det:
        out += " is " + det
    return out + "."


def wrap(line):
    """`line` broken after commas to fit 79 columns, as in array.m."""
    if len(line) <= 79:
        return line
    out, cur = [], ""
    for piece in re.split(r"(?<=, )", line):
        if len(cur) + len(piece) > 79 and cur:
            out.append(cur.rstrip())
            cur = "    " + piece
        else:
            cur += piece
    out.append(cur.rstrip())
    return "\n".join(out)


def main():
    source, output = sys.argv[1], sys.argv[2]
    lines = open(source).read().split("\n")
    # The stub's own comment at the top is replaced by the header.
    while lines and (lines[0].startswith("%") or not lines[0].strip()):
        lines.pop(0)
    items, cur = [], []
    for line in lines:
        if not cur and (not line.strip() or not line.startswith(":- ")):
            items.append(line)
            continue
        cur.append(line)
        if line.rstrip().endswith("."):
            items.append("\n".join(cur))
            cur = []
    # Which arguments of each predicate and function are arrays, by name,
    # arity and kind.
    arrays = {}
    for item in items:
        if item.startswith((":- pred ", ":- func ")):
            kind, name, arguments, result, _ = parse(item)
            types = [split_top(a, ":")[0].strip() if "::" in a else a
                     for a in arguments]
            arrays[(kind, name, len(arguments))] = [
                t.startswith("array(") for t in types]
    out = [HEADER]
    for item in items:
        if item in INSTS:
            out.append(INSTS[item])
            continue
        if not item.startswith((":- pred ", ":- func ", ":- mode ")):
            out.append(item)
            continue
        if item.startswith(":- mode ") and "==" in item:
            out.append(item)  # a mode definition
            continue
        kind, name, arguments, result, det = parse(item)
        modes = None
        if kind in ("pred", "func") and any("::" in a for a in arguments + [result or ""]):
            types = [a.split("::")[0].strip() for a in arguments]
            modes = [a.split("::")[1].strip() for a in arguments]
            rtype = rmode = None
            if result is not None:
                rtype, rmode = [x.strip() for x in result.split("::")]
            out.append(wrap(text(kind, name, types, rtype, "")))
            kind = "mode"
            arguments, result = modes, rmode
            modes = True
        if kind != "mode":
            out.append(item)
            continue
        key = ("func" if result is not None else "pred", name, len(arguments))
        which = arrays.get(key, [])
        # The arrays taken `in`, each of which may be taken `array_ui`
        # instead: every mix, those with more `array_ui` first.
        ins = [i for i, a in enumerate(arguments)
               if i < len(which) and which[i] and a == "in"]
        mixes = sorted(itertools.product([True, False], repeat=len(ins)),
                       key=lambda mix: -sum(mix))
        for mix in mixes:
            if not any(mix):
                continue
            ui = list(arguments)
            for i, u in zip(ins, mix):
                if u:
                    ui[i] = "array_ui"
            out.append(wrap(text("mode", name, ui, result, det)))
        if modes or ins:
            out.append(wrap(text("mode", name, arguments, result, det)))
        else:
            out.append(item)
    open(output, "w").write("\n".join(out).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
