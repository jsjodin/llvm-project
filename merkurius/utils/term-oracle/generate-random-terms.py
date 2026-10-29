#!/usr/bin/env python3
"""Write random Mercury items, for compare-terms.py.

The items come from a small grammar of Mercury's surface syntax: names,
variables, literals of every kind, compound terms, lists, tuples,
higher-order terms, backquoted operators, and prefix and infix operators
from the builtin table, with and without parentheses. Some items are not
valid; what matters is that merkc and Mercury's term parser agree on each.
The output depends only on --seed.
"""

import argparse
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "..", "..", "lib", "Parser", "operators.def")

ROWS = [
    (n.encode().decode("unicode_escape"), s)
    for n, s in re.findall(
        r'^MERK_OPERATOR\("((?:[^"\\]|\\.)*)", (\w+), \d+\)', open(TABLE).read(), re.M
    )
]
INFIX = sorted({n for n, s in ROWS if s in ("XFX", "XFY", "YFX")})
PREFIX = sorted({n for n, s in ROWS if s in ("FX", "FY")})
BINARY_PREFIX = sorted({n for n, s in ROWS if s == "FXY"})
COMMON_INFIX = ["=", ",", ";", ":-", "+", "-", "*", "/", "->", "::", ".", "^",
                "==", "\\=", "=<", "++", "&", ":=", "=>", "<=>", "-->", "--->"]

NAMES = ["a", "b", "foo", "x_1", "[]", "{}", "'A'", "'hello world'", "'it''s'",
         "'\\n'", "''", "'+'", "'-'", "';'", "'[]'", "'{}'", "'%'", "'.'",
         "'\\x41\\'", "'\\u00e9'", "'é'"]
VARIABLES = ["X", "Y", "_", "_Z", "Var_1", "!X", "!.X", "!:X"]
INTEGERS = ["0", "1", "42", "1_000", "0x1F", "0xff_u8", "0b101", "0o17", "0'a",
            "0' ", "0'\\\\", "0''", "7u", "127i8", "128i8", "65535u16",
            "9223372036854775807", "9223372036854775808",
            "18446744073709551616u", "123456789012345678901234567890"]
FLOATS = ["1.5", "0.1", "1.0e10", "2.5E-3", "1_000.000_1", "1.5_e10", "1e5",
          "1.0e309", "1.0e-400", "3.0"]
STRINGS = ['""', '"s"', '"a\\nb"', '"say ""hi"""', '"\\x41\\\\101\\"',
           '"\\u00e9"', '"%"', '"tab\\t"']


# A character code, or a quoted name or string with Mercury's escapes.
ESCAPE = r"\\x[0-9a-fA-F]+\\|\\[0-7]+\\|\\.|"
QUOTED = re.compile(
    r"0'(?:\\\\|.)"
    + r"|'(?:''|" + ESCAPE + r"[^'\\])*'"
    + r'|"(?:""|' + ESCAPE + r'[^"\\])*"'
)


def token_boundaries(text):
    """The positions of the spaces in `text` that are outside quotes."""
    boundaries, j = [], 0
    while j < len(text):
        match = QUOTED.match(text, j)
        if match:
            j = match.end()
            continue
        if text[j] == " ":
            boundaries.append(j)
        j += 1
    return boundaries


class Generator:
    def __init__(self, rng):
        self.rng = rng

    def pick(self, choices):
        return self.rng.choice(choices)

    def term(self, depth):
        r = self.rng.random()
        if depth <= 0 or r < 0.3:
            return self.primary(depth)
        if r < 0.65:
            op = self.pick(COMMON_INFIX if self.rng.random() < 0.6 else INFIX)
            left, right = self.term(depth - 1), self.term(depth - 1)
            if op == ".":
                # Glued, as "a . b" would end the item, and with a number
                # parenthesized, as "1._" would be a malformed float.
                if left[-1].isdigit():
                    left = f"({left})"
                return f"{left}.{right}"
            return f"{left} {op} {right}"
        if r < 0.8:
            op = self.pick(PREFIX)
            # Graphic operators may touch their operand; names may not.
            space = " " if op[0].isalpha() or self.rng.random() < 0.8 else ""
            return f"{op}{space}{self.term(depth - 1)}"
        if r < 0.85:
            return f"{self.pick(BINARY_PREFIX)} [{self.var()}] {self.term(depth - 1)}"
        if r < 0.9:
            return f"{self.term(depth - 1)} `{self.pick(['f', 'max', 'int.max', 'F', 'mod'])}` {self.term(depth - 1)}"
        return f"({self.term(depth - 1)})"

    def var(self):
        return self.pick(VARIABLES[:5])

    def arguments(self, depth):
        n = self.rng.choice([1, 1, 2, 2, 3])
        return ", ".join(self.term(depth - 1) for _ in range(n))

    def primary(self, depth):
        r = self.rng.random()
        if r < 0.2:
            return self.pick(NAMES)
        if r < 0.35:
            return self.pick(VARIABLES)
        if r < 0.45:
            sign = self.pick(["", "", "-", "- ", "-(", "'-' "])
            literal = self.pick(INTEGERS + FLOATS)
            return f"{sign}{literal})" if sign == "-(" else sign + literal
        if r < 0.5:
            return self.pick(STRINGS)
        if r < 0.55:
            return self.pick(["$line", "$file", "$module", "$pred"])
        if depth <= 0:
            return self.pick(NAMES)
        if r < 0.7:
            functor = self.pick(["f", "g", "'+'", "'-'", "[]", "{}", "X", "\"s\"", "1", "(f)"])
            return f"{functor}({self.arguments(depth)})"
        if r < 0.82:
            if self.rng.random() < 0.15:
                return "[]"
            tail = f" | {self.term(depth - 1)}" if self.rng.random() < 0.3 else ""
            return f"[{self.arguments(depth)}{tail}]"
        if r < 0.9:
            return f"{{{self.arguments(depth)}}}" if self.rng.random() < 0.85 else "{}"
        if r < 0.95:
            # An operator used as a constant.
            op = self.pick(INFIX + PREFIX)
            return self.pick([f"({op})", f"f({op})", f"[{op}]", op])
        return f"({self.term(depth - 1)})({self.arguments(depth)})"

    def item(self):
        text = self.term(self.rng.choice([2, 3, 4, 5]))
        # Syntax errors: a token out of place, or the item cut short. Both
        # happen only between tokens, as lexical errors are the lexer's
        # business, and mmc recovers from them differently.
        r = self.rng.random()
        cuts = token_boundaries(text)
        if r < 0.03 and cuts:
            cut = self.pick(cuts)
            text = text[:cut] + self.pick([" )", " ,", " |", " (", " ]", " `"]) + text[cut:]
        elif r < 0.05 and cuts:
            text = text[: self.pick(cuts)]
        # A space, so that an item ending in a graphic name still ends.
        return text + " .\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    generator = Generator(random.Random(args.seed))
    for _ in range(args.count):
        print(generator.item(), end="")


if __name__ == "__main__":
    main()
