#!/usr/bin/env python3
"""Compare merkc's type checking and inference with Mercury's on modules.

Each module is checked by `merkc -fc1 -fdebug-dump-types` and by
`mmc --errorcheck-only --infer-all`. mmc's inference is little used, and
its checking of declared types is the better referee, so differences are
settled by mmc with types declared where they can be. The script exits
with status 1 and reports each difference:

- Type errors. A module is to have none, unless --allow-errors is given.
  When mmc reports none, merkc must report none. Otherwise mmc reports the
  errors of the first pass of inference that has any, and no more, and may
  report errors that only the first pass sees (see the design document's
  register). So each line where only one of them reports an error is
  settled by mmc with the other predicates' and functions' signatures, as
  merkc inferred them, declared: mmc must then report an error there just
  when merkc does, and none where merkc does not.
- Inferred signatures. When neither reports a type error, each predicate
  or function whose type both infer must have the same signature, up to
  the names of its type variables, and each must infer the same ones.
  Where they differ, the module with mmc's signatures declared is checked
  by mmc: if mmc rejects its own inference, that is noted, and merkc's
  must be at least as general.
- The "unresolved polymorphism" warnings, when neither reports an error.
- Consistency. With merkc's inferred signatures declared, merkc must give
  each variable of each clause the type that inference gave it.
- Soundness. When merkc reports no error, its typed version must pass
  both merkc and mmc: the module with merkc's inferred signatures declared,
  each variable of a clause qualified with its type (`X = (X : T)`), and
  each overloaded name qualified with the module of its chosen symbol.

A line with the comment `mmc-differs` is left out: errors on it, and the
signature of a predicate or function with a clause starting on it. A test
of what merkc does there belongs elsewhere.

mmc's other messages, such as mode and determinism errors, are not
compared, since merkc does not check those yet.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Types, as merkc and mmc print them.


def tokenize(text):
    tokens = []
    for match in re.finditer(
        r"\s*('(?:[^'\\]|\\.)*'(?:\.\w+)*|[A-Za-z_][\w.]*(?:\.'(?:[^'\\]|\\.)*')?|[(){},=])",
        text,
    ):
        tokens.append(match.group(1))
    return tokens


class Parser:
    """Parses a type or a signature into a tuple tree: ("var", name),
    ("named", name, args), ("tuple", args) or ("ho", kind, purity, args,
    result)."""

    def __init__(self, text):
        self.tokens = tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if expected is not None and token != expected:
            raise ValueError("expected %r, got %r in %r" % (expected, token, self.tokens))
        self.i += 1
        return token

    def args(self):
        self.take("(")
        result = [self.eq()]
        while self.peek() == ",":
            self.take(",")
            result.append(self.eq())
        self.take(")")
        return result

    def eq(self):
        t = self.primary()
        if self.peek() == "=":
            self.take("=")
            result = self.primary()
            if t[0] != "ho" or t[1] != "func" or t[4] is not None:
                raise ValueError("unexpected = in %r" % self.tokens)
            t = ("ho", "func", t[2], t[3], result)
        return t

    def primary(self):
        token = self.take()
        purity = "pure"
        if token in ("impure", "semipure"):
            purity, token = token, self.take()
        if token in ("pred", "func"):
            arguments = []
            if self.peek() == "(":
                arguments = self.args()
            elif self.peek() not in (None, ")", ",", "="):
                # mmc writes `(func V_2) = T` for a function of one argument.
                arguments = [self.primary()]
            return ("ho", token, purity, tuple(arguments), None)
        if token == "(":
            t = self.eq()
            self.take(")")
            return t
        if token == "{":
            if self.peek() == "}":
                self.take("}")
                return ("tuple", ())
            elements = [self.eq()]
            while self.peek() == ",":
                self.take(",")
                elements.append(self.eq())
            self.take("}")
            return ("tuple", tuple(elements))
        if re.match(r"[A-Z_]", token):
            return ("var", token)
        name = token
        if name == "character":
            name = "char"
        arguments = self.args() if self.peek() == "(" else []
        return ("named", name, tuple(arguments))


def canonical(t, names):
    """The type as text, its variables renamed in order of occurrence."""
    if t[0] == "var":
        if t[1] not in names:
            names[t[1]] = "V%d" % len(names)
        return names[t[1]]
    if t[0] == "named":
        if not t[2]:
            return t[1]
        return "%s(%s)" % (t[1], ", ".join(canonical(a, names) for a in t[2]))
    if t[0] == "tuple":
        return "{%s}" % ", ".join(canonical(a, names) for a in t[1])
    _, kind, purity, arguments, result = t
    text = "" if purity == "pure" else purity + " "
    text += kind
    if arguments:
        text += "(%s)" % ", ".join(canonical(a, names) for a in arguments)
    if kind == "func":
        text = "(%s = %s)" % (text, canonical(result, names))
    return text


def parse_signature(text):
    """`pred name(T1, T2)` or `func name(T1) = T2` as (key, canonical text),
    the key being (kind, unqualified name, arity)."""
    match = re.match(r"(pred|func)\s+(.*)$", text.strip().rstrip("."))
    kind, rest = match.group(1), match.group(2)
    name_match = re.match(r"((?:'(?:[^'\\]|\\.)*'|[\w]+)(?:\.(?:'(?:[^'\\]|\\.)*'|\w+))*)", rest)
    full = name_match.group(1)
    rest = rest[len(full):]
    name = re.split(r"\.(?=(?:[^']*'[^']*')*[^']*$)", full)[-1]
    parser = Parser(rest)
    arguments = []
    if parser.peek() == "(":
        arguments = parser.args()
    names = {}
    text = "%s %s(%s)" % (kind, name, ", ".join(canonical(a, names) for a in arguments))
    if kind == "func":
        parser.take("=")
        text += " = " + canonical(parser.eq(), names)
    if parser.peek() is not None:
        raise ValueError("trailing tokens in %r" % rest)
    return (kind, name, len(arguments)), text


# mmc's output.


def mmc_messages(output, filename):
    """mmc's messages as (line, text) pairs, a message's lines joined; a
    message without a line has line 0."""
    messages = []
    prefix = re.compile(re.escape(filename) + r":(\d+): (.*)$")
    for raw in output.splitlines():
        match = prefix.match(raw)
        if match:
            line, text = int(match.group(1)), match.group(2)
            if text.startswith("  ") and messages and messages[-1][0] == line:
                messages[-1][1] += " " + text.strip()
            else:
                messages.append([line, text.strip()])
        elif raw.startswith("  ") and messages:
            messages[-1][1] += " " + raw.strip()
        elif raw.strip() and not raw.startswith("For more information"):
            messages.append([0, raw.strip()])
    return messages


TYPE_ERROR = re.compile(
    r"type error|ambiguous overloading|undefined (symbol|predicate|function|type)|"
    r"type inference iteration limit|without corresponding|wrong number of arguments|syntax error",
    re.IGNORECASE,
)


class MmcCrash(Exception):
    pass


def run_mmc(mmc, path, module):
    with tempfile.TemporaryDirectory() as tmp:
        copy = os.path.join(tmp, module + ".m")
        shutil.copyfile(path, copy)
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(mmc) + os.pathsep + env.get("PATH", "")
        result = subprocess.run(
            [mmc, "--errorcheck-only", "--infer-all", module + ".m"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        messages = mmc_messages(output, module + ".m")
        if "Uncaught Mercury exception" in output:
            raise MmcCrash(output.strip().splitlines()[-1])
    errors, warnings, signatures = set(), set(), {}
    # "In clause for ...:" introduces the message after it.
    i = 0
    while i < len(messages):
        line, text = messages[i]
        if text.startswith("In clause for") or text.startswith("In clause of"):
            body = text
            if i + 1 < len(messages) and messages[i + 1][0] == line:
                pass
            i += 1
        else:
            body = text
            i += 1
        if body.startswith("Inferred :- mode "):
            pass
        elif body.startswith("Inferred :- "):
            raw = body[len("Inferred :- "):].rstrip(".")
            key, signature = parse_signature(raw)
            signatures[key] = (line, signature, raw)
        elif TYPE_ERROR.search(body):
            errors.add(line)
        elif "unresolved polymorphism" in body:
            warnings.add(line)
    return errors, warnings, signatures


# merkc's output.


def run_merkc(merkc, path):
    result = subprocess.run(
        [merkc, "-fc1", "-fdebug-dump-types", os.path.basename(path)],
        cwd=os.path.dirname(os.path.abspath(path)),
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1) or "Assertion" in result.stderr:
        sys.exit("merkc failed on %s:\n%s" % (path, result.stdout + result.stderr))
    global last_dump, clause_starts
    last_dump = result.stdout
    clause_starts = []
    errors, warnings, signatures, clause_lines = set(), set(), {}, {}
    name = re.escape(os.path.basename(path))
    for raw in (result.stdout + result.stderr).splitlines():
        match = re.match(name + r":(\d+):\d+: (error|warning): (.*)$", raw)
        if match:
            line, severity, text = int(match.group(1)), match.group(2), match.group(3)
            if severity == "error":
                errors.add(line)
            elif "unresolved polymorphism" in text:
                warnings.add(line)
            continue
        match = re.match(r"clause at (\d+):\d+ of (?:pred|func) (.*)/(\d+)(?:: (inferred )?(.*))?$", raw)
        if match:
            line = int(match.group(1))
            clause_starts.append((line, match.group(2).split(".")[-1], int(match.group(3))))
            if match.group(4) and match.group(5):
                key, signature = parse_signature(match.group(5))
                signatures[key] = (line, signature, match.group(5))
                clause_lines.setdefault(key, set()).add(line)
    return errors, warnings, signatures, clause_lines


def with_declarations(text, declarations):
    """The module with these declarations after `:- implementation.`."""
    lines = ["\n:- %s." % d for d in declarations]
    return re.sub(r"^:- implementation\.\n", lambda m: m.group(0) + "".join(lines) + "\n", text, count=1, flags=re.M)


def check_variant(text, module, merkc, mmc, which):
    """The type errors merkc and mmc (in `which`) report on a variant."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, module + ".m")
        open(path, "w").write(text)
        result = {}
        if "merkc" in which:
            result["merkc"] = run_merkc(merkc, path)[0]
        if "mmc" in which:
            result["mmc"] = run_mmc(mmc, path, module)[0]
        return result


def module_of(text):
    return re.search(r"^:- module ([\w.]+)\.", text, re.M).group(1)


last_dump = ""
clause_starts = []


def clause_end(text, j):
    """The offset of the end token of the clause starting at offset j, and
    the offset of its top-level `:-`, or None."""
    neck, depth = None, 0
    while j < len(text):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "%":
            j = text.find("\n", j)
            j = len(text) if j < 0 else j
        elif ch in "\"'`":
            j += 1
            while j < len(text) and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
        elif ch == "0" and text[j + 1 : j + 2] == "'":
            j += 2
        elif text.startswith(":-", j) and neck is None and depth == 0:
            neck = j
            j += 1
        elif ch == "." and (j + 1 == len(text) or text[j + 1] in " \t\r\n%"):
            if j == 0 or not text[j - 1] in "#$&*+-./:<=>?@^~\\":
                return j, neck
        j += 1
    return None, neck


def typed_version(text):
    """The module with merkc's choices written in, from the last dump: each
    variable of a clause qualified with its type, and each overloaded
    name that is an identifier qualified with its module. Variables whose
    types are not determined, and DCG clauses, are left alone."""
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def offset(line, column):
        return offsets[line - 1] + column - 1

    edits = []
    clause, types = None, {}

    def finish():
        if clause is None:
            return
        end, neck = clause_end(text, clause)
        if end is None or "-->" in text[clause:end]:
            return
        goals = [
            "%s = (%s : %s)" % (v, v, t)
            for v, t in types.items()
            if t is not None and "T_" not in t and "<" not in t
        ]
        if not goals:
            return
        if neck is not None and neck < end:
            edits.append((neck + 2, 0, " ("))
            edits.append((end, 0, "), " + ", ".join(goals)))
        else:
            edits.append((end, 0, " :- " + ", ".join(goals)))

    for raw in last_dump.splitlines():
        match = re.match(r"clause at (\d+):(\d+) of ", raw)
        if match:
            finish()
            clause, types = offset(int(match.group(1)), int(match.group(2))), {}
            continue
        match = re.match(r"  ([A-Z]\w*): (.*)$", raw)
        if match:
            # A name with two variables (in different scopes) is skipped.
            types[match.group(1)] = None if match.group(1) in types else match.group(2)
            continue
        match = re.match(r"  (\d+):(\d+) ([a-z]\w*) -> (pred|func|constructor) (\w+)\.(\w+)/\d+$", raw)
        if match and match.group(3) == match.group(6):
            at = offset(int(match.group(1)), int(match.group(2)))
            name = match.group(3)
            if text[at : at + len(name)] == name and (at == 0 or text[at - 1] not in ".'"):
                edits.append((at, len(name), match.group(5) + "." + name))
    finish()
    for at, length, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        text = text[:at] + replacement + text[at + length :]
    return text


def clause_types(dump):
    """Each clause's variables' types, in the order of the clauses."""
    clauses = []
    for raw in dump.splitlines():
        if raw.startswith("clause at "):
            clauses.append({})
        elif clauses:
            match = re.match(r"  ([A-Z_]\w*|![.:]?\w+): (.*)$", raw)
            if match:
                clauses[-1].setdefault(match.group(1), []).append(match.group(2))
    return clauses


def check_consistent(text, merkc, declarations):
    """merkc's inferred signatures, declared, give the same types."""
    original = last_dump
    inferred = clause_types(original)
    with tempfile.TemporaryDirectory() as tmp:
        variant = os.path.join(tmp, module_of(text) + ".m")
        open(variant, "w").write(with_declarations(text, declarations))
        errors = run_merkc(merkc, variant)[0]
        declared = clause_types(last_dump)
    if errors:
        return ["merkc rejects its inferred signatures, declared, at lines %s of the variant" % sorted(errors)]
    differences = []
    lines = re.findall(r"^clause at (\d+):\d+ of ", original, re.M)
    for k, (a, b) in enumerate(zip(inferred, declared)):
        for var in sorted(set(a) | set(b)):
            if a.get(var) != b.get(var):
                differences.append(
                    "clause at line %s: inference gives %s the type %s, the declared signatures %s"
                    % (lines[k], var, a.get(var), b.get(var))
                )
    if len(inferred) != len(declared):
        differences.append("the declared signatures give %d clauses, inference %d" % (len(declared), len(inferred)))
    return differences


def check_typed_by_merkc(text, merkc, signatures):
    """The consistency check, and merkc's check of its typed version."""
    declared_keys = set(re.findall(r"^:- (pred|func) (\w+)", text, re.M))
    declarations = [raw for key, (line, _, raw) in signatures.items() if (key[0], key[1]) not in declared_keys]
    typed = typed_version(text)
    differences = check_consistent(text, merkc, declarations)
    errors = check_variant(with_declarations(typed, declarations), module_of(text), merkc, None, ("merkc",))
    if errors["merkc"]:
        differences.append("merkc rejects its typed version, at lines %s" % sorted(errors["merkc"]))
    return differences


def check_declared(path, text, merkc, mmc, signatures):
    """merkc's inferred signatures, declared, and its other choices written
    in, must pass both checkers."""
    typed = typed_version(text)
    declared_keys = set(re.findall(r"^:- (pred|func) (\w+)", text, re.M))
    declarations = [raw for key, (line, _, raw) in signatures.items() if (key[0], key[1]) not in declared_keys]
    if not declarations:
        return []
    differences = check_consistent(text, merkc, declarations)
    variant = with_declarations(typed, declarations)
    errors = check_variant(variant, module_of(text), merkc, mmc, ("merkc", "mmc"))
    for checker in ("merkc", "mmc"):
        if errors[checker]:
            lines = variant.splitlines()
            differences.append(
                "%s rejects merkc's typed version, at:\n%s"
                % (checker, "\n".join("    " + lines[l - 1] for l in sorted(errors[checker]) if 0 < l <= len(lines)))
            )
    if os.environ.get("TYPE_ORACLE_KEEP"):
        open(os.path.join(os.environ["TYPE_ORACLE_KEEP"], os.path.basename(path)), "w").write(variant)
    return differences


def more_general(general, specific):
    """Whether type tree `specific` is an instance of `general`."""
    binding = {}

    def match(g, s):
        if g[0] == "var":
            if g[1] in binding:
                return binding[g[1]] == s
            binding[g[1]] = s
            return True
        if g[0] != s[0]:
            return False
        if g[0] == "named":
            return g[1] == s[1] and len(g[2]) == len(s[2]) and all(map(match, g[2], s[2]))
        if g[0] == "tuple":
            return len(g[1]) == len(s[1]) and all(map(match, g[1], s[1]))
        return (
            g[1:3] == s[1:3]
            and len(g[3]) == len(s[3])
            and all(map(match, g[3], s[3]))
            and ((g[4] is None) == (s[4] is None))
            and (g[4] is None or match(g[4], s[4]))
        )

    return match(general, specific)


def signature_tree(raw):
    """A signature's argument types, with a function's result, as a tuple."""
    rest = re.sub(r"^(pred|func)\s+(?:'(?:[^'\\]|\\.)*'|[\w.]+)", "", raw.strip())
    parser = Parser(rest)
    arguments = parser.args() if parser.peek() == "(" else []
    if parser.peek() == "=":
        parser.take("=")
        arguments.append(parser.eq())
    return ("tuple", tuple(arguments))


def adjudicate(path, text, merkc, mmc, key, signatures, mmc_signatures):
    """Settles a difference in inferred signatures by mmc's checking of
    declared types: true if mmc rejects its own inference, declared, and
    merkc's is at least as general as one mmc accepts."""
    declared_keys = set(re.findall(r"^:- (pred|func) (\w+)", text, re.M))
    theirs = [raw for k, (_, _, raw) in mmc_signatures.items() if (k[0], k[1]) not in declared_keys]
    errors = check_variant(with_declarations(text, theirs), module_of(text), merkc, mmc, ("mmc",))["mmc"]
    if not errors:
        # Both may be valid; merkc's must then be the more general.
        return more_general(signature_tree(signatures[key][2]), signature_tree(mmc_signatures[key][2])) and not more_general(
            signature_tree(mmc_signatures[key][2]), signature_tree(signatures[key][2])
        )
    print(
        "%s: note: mmc infers %s %s/%d as %s, which mmc rejects when declared; merkc infers %s"
        % (path, key[0], key[1], key[2], mmc_signatures[key][1], signatures[key][1])
    )
    return True


def owner_of(line, starts, declarations=None):
    """The predicate or function, as (name, arity), whose clause the line
    is in, or with `declarations`, the module's text, whose declaration it
    is, which is that of the next clause."""
    if declarations is not None:
        lines = declarations.splitlines()
        if 0 < line <= len(lines) and re.match(r":- (pred|func) ", lines[line - 1]):
            start = min((start for start in starts if start[0] > line), default=None)
            return None if start is None else (start[1], start[2])
    start = max((start for start in starts if start[0] <= line), default=None)
    return None if start is None else (start[1], start[2])


def callers(owner, starts, dump):
    """The predicates and functions with a clause that calls this one."""
    result = set()
    for match in re.finditer(r"^  (\d+):\d+ \S+ -> (?:pred|func) [\w.]*?(\w+)/(\d+)$", dump, re.M):
        if (match.group(2), int(match.group(3))) == owner:
            result.add(owner_of(int(match.group(1)), starts))
    return result


def recursive(owner, starts, dump):
    """Whether a clause of the predicate or function calls it."""
    return owner in callers(owner, starts, dump)


def settle(text, mmc, owner, errors, signatures, starts, dump, own=False):
    """Whether mmc reports an error in a clause of the predicate or function
    with the signatures merkc inferred declared, but its own unless `own`;
    None if mmc then reports an error in one where merkc does not, other
    than its callers, which its first pass of inference may not type."""
    declarations = [raw for key, (_, _, raw) in signatures.items() if own or (key[1], key[2]) != owner]
    implementation = text[: text.index(":- implementation.")].count("\n") + 1
    shift = len(declarations) + 1
    variant = with_declarations(text, declarations)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, module_of(text) + ".m")
        open(path, "w").write(variant)
        reported = run_mmc(mmc, path, module_of(text))[0]
    lines = {l - shift if l > implementation + shift else l for l in reported} - {0}
    owners = {owner_of(l, starts) for l in lines}
    if owners - {owner_of(l, starts) for l in errors} - {owner} - (set() if own else callers(owner, starts, dump)):
        return None
    return owner in owners


def compare(path, merkc, mmc, allow_errors=False):
    text = open(path).read()
    module = re.search(r"^:- module ([\w.]+)\.", text, re.M).group(1)
    exempt = {i + 1 for i, line in enumerate(text.splitlines()) if "mmc-differs" in line}
    try:
        mmc_errors, mmc_warnings, mmc_signatures = run_mmc(mmc, path, module)
    except MmcCrash as crash:
        # Only merkc's own checks are left.
        print("%s: note: mmc crashes (%s); its verdicts are not compared" % (path, crash))
        errors, warnings, signatures, clause_lines = run_merkc(merkc, path)
        if errors:
            return [] if allow_errors else ["merkc reports type errors, at lines %s" % sorted(errors)]
        return check_typed_by_merkc(text, merkc, signatures)
    errors, warnings, signatures, clause_lines = run_merkc(merkc, path)
    differences = []
    missed_fixpoint = False
    if 0 in mmc_errors and not errors:
        # mmc's iteration limit. With merkc's signatures declared, mmc must
        # accept the module: mmc then missed a fixpoint (see the register).
        declarations = [raw for _, _, raw in signatures.values()]
        with tempfile.TemporaryDirectory() as tmp:
            variant = os.path.join(tmp, module + ".m")
            open(variant, "w").write(with_declarations(text, declarations))
            declared_errors = run_mmc(mmc, variant, module)[0]
        if declared_errors:
            differences.append("mmc reports a type error without a line, and merkc reports none")
        else:
            print("%s: note: mmc does not reach a fixpoint that merkc does, and accepts merkc's signatures declared" % path)
            mmc_errors = mmc_errors - {0}
            missed_fixpoint = True
    if not allow_errors and errors - exempt:
        differences.append("merkc reports type errors, at lines %s" % sorted(errors - exempt))
    if not allow_errors and mmc_errors - exempt:
        differences.append("mmc reports type errors, at lines %s" % sorted(mmc_errors - exempt))
    starts = list(clause_starts)
    dump = last_dump
    if not mmc_errors - exempt and errors - exempt:
        differences.append("merkc reports type errors that mmc does not, at lines %s" % sorted(errors - exempt))
    elif mmc_errors - exempt:
        # By predicate or function: one may report an ambiguity of its
        # clauses at the first, and the other at the one with the name.
        ours_owners = {owner_of(l, starts) for l in errors - exempt}
        mmc_owners = {owner_of(l, starts) for l in mmc_errors - exempt - {0}}
        for owner in sorted(ours_owners ^ mmc_owners, key=str):
            ours = owner in ours_owners
            verdict = settle(text, mmc, owner, errors - exempt, signatures, starts, dump)
            name = "%s/%d" % owner if owner else "the module"
            if verdict and not ours and recursive(owner, starts, dump) and owner in {(k[1], k[2]) for k in signatures}:
                # mmc's first pass may not know the predicate's own type at
                # its recursive calls; with merkc's declared, which must be
                # valid, there is then no error.
                if settle(text, mmc, owner, errors - exempt, signatures, starts, dump, own=True) is False:
                    print("%s: note: mmc reports a type error in %s that its first pass of inference sees, and none with its type declared" % (path, name))
                    continue
            if verdict is None:
                differences.append(
                    "%s: with merkc's other inferred signatures declared, mmc reports type errors where merkc does not" % name
                )
            elif verdict != ours:
                differences.append(
                    "%s: only %s reports a type error, even with merkc's other inferred signatures declared"
                    % (name, "merkc" if ours else "mmc")
                )
    # mmc's inference stops at its first pass with an error, even one on
    # a line left out, so its signatures are compared only without any.
    if not mmc_errors and not errors - exempt and not missed_fixpoint:
        for key in sorted(set(mmc_signatures) | set(signatures)):
            if clause_lines.get(key, set()) & exempt or (key in mmc_signatures and mmc_signatures[key][0] in exempt):
                continue
            ours = signatures.get(key, (None, None, None))[1]
            theirs = mmc_signatures.get(key, (None, None, None))[1]
            if ours != theirs:
                if ours and theirs and adjudicate(path, text, merkc, mmc, key, signatures, mmc_signatures):
                    continue
                differences.append(
                    "%s %s/%d: merkc infers %s, mmc %s" % (key[0], key[1], key[2], ours, theirs)
                )
        # mmc warns once for a predicate or function, at its declaration
        # or first clause, and merkc for each clause.
        ours = {owner_of(l, starts) for l in warnings - exempt}
        theirs = {owner_of(l, starts, declarations=text) for l in mmc_warnings - exempt}
        if ours != theirs:
            differences.append(
                "unresolved polymorphism: merkc warns in %s, mmc in %s"
                % (sorted("%s/%d" % o for o in ours if o), sorted("%s/%d" % o for o in theirs if o))
            )
    if not errors - exempt:
        differences += check_declared(path, text, merkc, mmc, signatures)
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merkc", required=True)
    parser.add_argument("--mmc", required=True)
    parser.add_argument("--allow-errors", action="store_true",
                        help="the modules have type errors on purpose")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    failed = False
    for path in args.files:
        for difference in compare(path, args.merkc, args.mmc, args.allow_errors):
            print("%s: %s" % (path, difference))
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
