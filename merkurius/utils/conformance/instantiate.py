#!/usr/bin/env python3
"""Instantiates a templated conformance test for one floating point type.

A template, TEST.mt, is a test written once for every floating point type
(docs/INPROGRESS.md, Sized floating point types): its `% FLOAT-TYPES:'
line names the types, float and float32 if it has none, and its text uses
these placeholders, which each type's instance replaces:

  @T@          the type's name: float, float32
  @S@          the literal suffix, after a literal's digits: 1.5@S@ is
               1.5 or 1.5f32
  @TM@         the modules to import for the type: float, or float and
               float32, so that float's names are there in every instance
  @MATH@       the math module of the type: math, math_float32
  @MATHM@      the math modules to import: math, or math and math_float32
  @FROM_INT@   the conversion of an int: float.float, float32.cast_from_int
  @TO_FLOAT@   the conversion to float: nothing for float, float32.to_float
  @FROM_FLOAT@ the conversion of a float: nothing, float32.from_float

An Erlang TRANSLATION of a template is one too, with the same
placeholders, and its module keeps its name.

and the module, `:- module run_x.', is run_x_float32 in float32's
instance, in a file of that name. A test's other modules may be templates
too, and keep their names (--keep-name): each type's instance is in a
directory of its own. io.write_float32 is io.write_@T@, and
string.float32_to_string is string.@T@_to_string.

For mmc (--mmc), which has no float32, a float32 instance uses the stand-in
modules in emulation/, which compute in C's float: a literal 1.5f32 becomes
(float32.lit("1.5")), outside strings and comments, in parentheses so that
-1.5f32 is still the negation of the literal, and io.write_float32 and
string.float32_to_string become the stand-in's. What mmc's program prints
is normalised with normalise(): the stand-in's float32_value(X) becomes X
as merkc's programs write a float32, 1.5f32.

  instantiate.py [--mmc] [--emulation] [--keep-name] TEMPLATE TYPE DIRECTORY

writes the instance to DIRECTORY, made if need be, as MODULE.m, and prints
its path. With --emulation it also copies the stand-in modules there.
"""

import argparse
import os
import re
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EMULATION = os.path.join(HERE, "emulation")

# Each type's replacements. float16 and bfloat16 join when they are built
# (step 3), with stand-ins of their own.
TYPES = {
    "float": {
        "T": "float", "S": "", "TM": "float", "MATH": "math",
        "MATHM": "math", "FROM_INT": "float.float", "TO_FLOAT": "",
        "FROM_FLOAT": "",
    },
    "float32": {
        "T": "float32", "S": "f32", "TM": "float, float32",
        "MATH": "math_float32", "MATHM": "math, math_float32",
        "FROM_INT": "float32.cast_from_int", "TO_FLOAT": "float32.to_float",
        "FROM_FLOAT": "float32.from_float",
    },
}

# The stand-ins each type's mmc instance needs.
EMULATED = {"float32": ["float32.m", "math_float32.m"]}

DEFAULT_TYPES = ["float", "float32"]


def float_types(text):
    """The types a template names on its `% FLOAT-TYPES:' line."""
    m = re.search(r"^% FLOAT-TYPES:(.*)$", text, re.M)
    types = m.group(1).split() if m else DEFAULT_TYPES
    for t in types:
        if t not in TYPES:
            raise ValueError("no floating point type " + t)
    return types


def declared_name(text):
    m = re.search(r"^:- module ([a-z][A-Za-z0-9_]*)\.", text, re.M)
    if not m:
        raise ValueError("no `:- module' declaration")
    return m.group(1)


def module_name(text, type_name):
    return declared_name(text) + "_" + type_name


def code_segments(text):
    """The text as (is_code, segment) pairs: strings, quoted names and
    comments are not code."""
    i, start, n = 0, 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            if i > start:
                yield True, text[start:i]
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            yield False, text[i:j + 1]
            i = start = j + 1
        elif c == "%":
            if i > start:
                yield True, text[start:i]
            j = text.find("\n", i)
            j = n if j < 0 else j
            yield False, text[i:j]
            i = start = j
        else:
            i += 1
    if start < n:
        yield True, text[start:]


LITERAL = re.compile(
    r"(?<![\w.])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(f32)\b")


def for_mmc(text, type_name):
    """The instance as mmc compiles it, with the stand-in modules."""
    if type_name not in EMULATED:
        return text
    out = []
    for is_code, segment in code_segments(text):
        if is_code:
            segment = LITERAL.sub(
                lambda m: '(%s.lit("%s"))' % (type_name, m.group(1)),
                segment)
            segment = segment.replace("io.write_" + type_name,
                                      type_name + ".write_" + type_name)
            segment = segment.replace("string.%s_to_string" % type_name,
                                      "%s.%s_to_string" % (type_name,
                                                          type_name))
        out.append(segment)
    return "".join(out)


def substitute(text, type_name):
    """The text with the type's replacements for the placeholders; for a
    translation into another language too, as an Erlang one, whose
    `-define(FLOAT_TYPE, @T@).' names the type to mercury_write.hrl."""
    for key, value in TYPES[type_name].items():
        text = text.replace("@" + key + "@", value)
    unknown = re.search(r"@[A-Z_]+@", text)
    if unknown:
        raise ValueError("unknown placeholder " + unknown.group(0))
    return text


def instantiate(text, type_name, mmc=False, rename=True):
    """The template's instance for the type, and its module's name."""
    if not rename:
        module = declared_name(text)
    else:
        module = module_name(text, type_name)
        text = re.sub(r"^(:- (?:end_)?module) [a-z][A-Za-z0-9_]*\.",
                      r"\1 " + module + ".", text, flags=re.M)
    text = substitute(text, type_name)
    if mmc:
        text = for_mmc(text, type_name)
    return text, module


def format_float32(x):
    """A float32 as merkc's programs write it with io.write: the fewest of
    6 to 9 digits that read back as the same float32, and the suffix, as
    runtime/write.cpp has it."""
    if x != x:
        return "nan"
    if x in (float("inf"), float("-inf")):
        return "infinity" if x > 0 else "-infinity"
    single = struct.unpack("f", struct.pack("f", x))[0]
    for p in range(6, 10):
        s = "%.*g" % (p, single)
        if struct.unpack("f", struct.pack("f", float(s)))[0] == single:
            break
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f32"


STAND_IN = re.compile(r"float32_value\(([-+0-9.eE]+|infinity|-infinity|nan)\)")


def normalise(output, type_name):
    """What mmc's program printed, as merkc's would print it."""
    if type_name != "float32":
        return output
    return STAND_IN.sub(lambda m: format_float32(float(
        m.group(1).replace("infinity", "inf"))), output)


def write_instance(template, type_name, directory, mmc=False,
                   emulation=False, rename=True):
    with open(template, encoding="utf-8") as f:
        text, module = instantiate(f.read(), type_name, mmc, rename)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, module + ".m")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if emulation:
        for name in EMULATED.get(type_name, []):
            shutil.copy(os.path.join(EMULATION, name), directory)
    return path, module


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mmc", action="store_true")
    parser.add_argument("--emulation", action="store_true")
    parser.add_argument("--keep-name", action="store_true")
    parser.add_argument("template")
    parser.add_argument("type", choices=sorted(TYPES))
    parser.add_argument("directory")
    args = parser.parse_args()
    try:
        path, _ = write_instance(args.template, args.type, args.directory,
                                 args.mmc, args.emulation,
                                 not args.keep_name)
    except ValueError as e:
        sys.exit("instantiate.py: %s: %s" % (args.template, e))
    print(path)


if __name__ == "__main__":
    main()
