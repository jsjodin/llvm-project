#!/usr/bin/env python3
"""Write the conformance suite's interaction tests: a feature in a context.

Interactions are where compilers break (docs/INPROGRESS.md, A conformance
suite, "Interactions"). Every pair of a feature and a context by hand is too
many, so this writes a test for each pair, from a template for each feature
and each context, into test/Conformance/interactions/<context>/. The tests
are checked in, so that a failure reproduces and the expectations, which
the oracle writes from mmc's output, are reviewed; this is rerun when a
list grows, and then the oracle:

  generate-interactions.py
  oracle.py --update test/Conformance/interactions

A feature is a type and two functions over it, which exercise the feature:

  :- type fdata ...
  :- func make(int) = fdata.     builds a value of the type
  :- func use(fdata) = string.   takes it apart, as text to print

A context moves values of fdata through itself, and prints what use makes
of them, for a few values of make's argument: a closure builds and
captures them, a type class method makes and uses them, polymorphic code
carries them, a nondet goal makes several, a try goal throws one, or the
feature is another module's. The context does nothing a feature might not
allow, as unifying two values: the feature does that if it is the
feature, in use.

Usage: generate-interactions.py [--out DIR] [--check]
  --check   exit with 1 if a test would change, writing nothing
"""

import argparse
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "test", "Conformance", "interactions")


class Feature:
    def __init__(self, name, manual, imports, interface, implementation,
                 summary, interface_imports=()):
        self.gpu = None
        self.name = name
        self.manual = manual
        self.imports = set(imports) | set(interface_imports)
        # What the interface's types need, when it is another module's.
        self.interface_imports = set(interface_imports)
        # What another module exports: the type, make and use, and any
        # instance; the rest is its implementation.
        self.interface = interface.strip("\n")
        self.implementation = implementation.strip("\n")
        self.summary = summary


class Context:
    def __init__(self, name, manual, imports, code, summary, flags=(),
                 plain=None, io=False, gpu=False):
        self.name = name
        self.manual = manual
        self.imports = set(imports)
        # Defines run(N) = String, from make and use; or with io,
        # run(N, String, !IO).
        self.code = code.strip("\n")
        self.summary = summary
        # merkc's flags, for an extension of Merkurius's.
        self.flags = list(flags)
        # For an extension, which mmc does not have, the run of the
        # translation that mmc runs instead: plain Mercury that prints the
        # same, as the context must not change what the feature computes.
        self.plain = plain.strip("\n") if plain else None
        self.io = io
        # Whether the context also runs on the GPU, in a -gpu.test.
        self.gpu = gpu


FEATURES = [
    Feature(
        "layouts", "Discriminated unions", ["int", "float", "string"],
        """
    % More functors with arguments than a pointer has tags, constants, and
    % arguments of every kind of word.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, string)
    ;       three(float, fdata, int)
    ;       four(char, int, int, int)
    ;       five(string)
    ;       six(int)
    ;       seven(fdata, fdata)
    ;       eight(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = seven(three(float(N) / 2.0, one(N), N * 3),
    ( if N mod 2 = 0 then
        eight(c0)
    else if N mod 3 = 0 then
        four('x', N, -N, N * N)
    else
        two(N, "s" ++ string.int_to_string(N))
    )).

use(c0) = "c0".
use(c1) = "c1".
use(one(I)) = "one(" ++ string.int_to_string(I) ++ ")".
use(two(I, S)) = "two(" ++ string.int_to_string(I) ++ ", " ++ S ++ ")".
use(three(F, D, I)) = "three(" ++ string.float_to_string(F) ++ ", " ++
    use(D) ++ ", " ++ string.int_to_string(I) ++ ")".
use(four(C, A, B, D)) = "four(" ++ string.char_to_string(C) ++ ", " ++
    string.int_to_string(A + B + D) ++ ")".
use(five(S)) = "five(" ++ S ++ ")".
use(six(I)) = "six(" ++ string.int_to_string(I) ++ ")".
use(seven(A, B)) = "seven(" ++ use(A) ++ ", " ++ use(B) ++ ")".
use(eight(D)) = "eight(" ++ use(D) ++ ")".
""",
        "a type of more functors than a pointer has tags", ["char"]),
    Feature(
        "flat-values", "Discriminated unions",
        ["int", "float", "string"],
        """
:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag,
    % which Merkurius holds by value.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, float).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    float(N) * 0.25).

use(fdata(C, wrapped(W), I, U, F)) =
    string.string(C) ++ " " ++ string.int_to_string(W) ++ " " ++
    string.int_to_string(int8.to_int(I)) ++ " " ++
    string.int_to_string(uint16.to_int(U)) ++ " " ++
    string.float_to_string(F).
""",
        "an enumeration, a notag and a record of numbers",
        ["int8", "uint16"]),
    Feature(
        "closures-as-data", "Calling higher-order terms",
        ["int", "string"],
        """
    % Functions as data, which capture the argument of make.
:- type fdata
    --->    fns(func(int) = int, func(int) = string).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = fns((func(X) = X * N + 1),
    (func(X) = string.int_to_string(X) ++ "/" ++ string.int_to_string(N))).

use(fns(F, G)) = string.int_to_string(apply(F, 10)) ++ " " ++ G(F(2)).
""",
        "closures stored in a term and called"),
    Feature(
        "type-classes", "Type classes", ["int", "string", "list"],
        """
:- typeclass shape(T) where [
    func area(T) = int,
    func name(T) = string
].

:- type fdata
    --->    rect(int, int)
    ;       square(int).

:- instance shape(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
:- instance shape(fdata) where [
    area(rect(W, H)) = W * H,
    area(square(S)) = S * S,
    name(rect(_, _)) = "rect",
    name(square(_)) = "square"
].

make(N) = ( if N mod 2 = 0 then square(N) else rect(N, N + 1) ).

use(D) = describe([D, square(1)]).

    % Through a constraint, so that the method is called by its dictionary.
:- func describe(list(T)) = string <= shape(T).
:- pragma no_inline(describe/1).

describe(Shapes) = string.join_list(", ",
    list.map((func(S) = name(S) ++ "=" ++ string.int_to_string(area(S))),
        Shapes)).
""",
        "a type class's methods called through a constraint"),
    Feature(
        "existential-types", "Existentially typed data types",
        ["int", "string", "list"],
        """
:- typeclass showable(T) where [
    func show(T) = string
].

:- instance showable(int).
:- instance showable(string).

    % A value of a hidden type, with its dictionary.
:- type fdata
    --->    some [T] boxed(T) => showable(T).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
:- instance showable(int) where [
    show(I) = "int " ++ string.int_to_string(I)
].
:- instance showable(string) where [
    show(S) = "string " ++ S
].

make(N) = D :-
    ( if N mod 2 = 0 then
        D = 'new boxed'(N * 7)
    else
        D = 'new boxed'(string.duplicate_char('z', N))
    ).

use(boxed(X)) = show(X).
""",
        "an existentially typed constructor and its dictionary"),
    Feature(
        "equivalence-types", "Equivalence types",
        ["int", "string"],
        """
:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).
""",
        "a type defined as another, generic one", ["list", "pair"]),
    Feature(
        "user-defined-equality", "User-defined equality and comparison",
        ["int", "string", "list"],
        """
    % A set as a list in any order, equal by its members.
:- type fdata
    --->    set(list(int))
    where equality is same_members.

:- pred same_members(fdata::in, fdata::in) is semidet.

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
same_members(A, B) :-
    promise_equivalent_solutions [SA, SB] (
        A = set(LA),
        B = set(LB),
        list.sort_and_remove_dups(LA, SA),
        list.sort_and_remove_dups(LB, SB)
    ),
    SA = SB.

make(N) = set(list.reverse(1 .. N) ++ [1]).

use(D) = S :-
    % Built first, so that the unification compares two values, by the
    % type's equality.
    Expected = set([3, 2, 1]),
    ( if D = Expected then
        Equal = "equal to [1, 2, 3]"
    else
        Equal = "not equal to [1, 2, 3]"
    ),
    promise_equivalent_solutions [Count] (
        D = set(L),
        list.sort_and_remove_dups(L, Members),
        Count = list.length(Members)
    ),
    S = string.int_to_string(Count) ++ " members, " ++ Equal.
""",
        "a type with an equality of its own", ["list"]),
    Feature(
        "field-access", "Field access functions", ["int", "string", "list"],
        """
:- type fdata
    --->    account(
                owner :: string,
                balance :: int,
                history :: list(int)
            ).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = account("owner" ++ string.int_to_string(N), N * 10, []).

use(A0) = S :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ history := [A0 ^ balance | A1 ^ history],
    S = A ^ owner ++ " " ++ string.int_to_string(A ^ balance) ++ " " ++
        string.join_list(",", list.map(string.int_to_string, A ^ history)).
""",
        "field access and update functions", ["list"]),
    Feature(
        "determinism", "Determinism categories",
        ["int", "string", "list", "solutions"],
        """
:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = 1 .. N + 2.

use(L) = S :-
    solutions(even_member(L), Evens),
    ( if first_over(L, 2, F) then
        First = string.int_to_string(F)
    else
        First = "none"
    ),
    S = "evens " ++ string.join_list(",",
        list.map(string.int_to_string, Evens)) ++ ", first over 2 " ++ First.

:- pred even_member(list(int)::in, int::out) is nondet.

even_member(L, X) :-
    list.member(X, L),
    X mod 2 = 0.

:- pred first_over(list(int)::in, int::in, int::out) is semidet.

first_over([X | Xs], Limit, F) :-
    ( if X > Limit then
        F = X
    else
        first_over(Xs, Limit, F)
    ).
""",
        "nondet and semidet code", ["list"]),
    Feature(
        "unique-modes", "Unique modes", ["int", "string", "list", "array"],
        """
:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = N + 3.

use(Size) = string.join_list(" ",
    list.map(string.int_to_string, array.to_list(squares(0, A0)))) :-
    array.init(Size, 0, A0).

    % Updates the array in place, as its modes allow.
:- func squares(int, array(int)) = array(int).
:- mode squares(in, array_di) = array_uo is det.

squares(I, A0) = A :-
    ( if I < array.size(A0) then
        array.set(I, I * I, A0, A1),
        A = squares(I + 1, A1)
    else
        A = A0
    ).
""",
        "an array updated in place in unique modes"),
    Feature(
        "try-goals", "Exception handling",
        ["int", "string", "exception"],
        """
:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.
""",
        """
make(N) = N.

use(N) = S :-
    promise_equivalent_solutions [S] (
        ( try []
            risky(N, R)
        then
            S = "ok " ++ string.int_to_string(R)
        catch Message ->
            S = "caught " ++ Message
        )
    ).

:- pred risky(int::in, int::out) is det.

risky(N, R) :-
    ( if N mod 2 = 0 then
        throw("even " ++ string.int_to_string(N))
    else
        R = N * 3
    ).
""",
        "a try goal that catches a thrown string"),
]

# The features that a GPU process may have, as flat values: each feature's
# variant for the GPU, whose use is an int, a digest of the value, as a GPU
# process makes no strings. (imports, interface, implementation) Closures,
# existential types, user-defined equality over lists and try goals make
# values that are not flat; type classes wait for a GPU process's method
# calls ("dispatches on a type class, which cannot be done in a GPU process
# yet").
GPU_VARIANTS = {
    "layouts": (["int", "float"], """
    % More functors with arguments than a pointer has tags, and constants,
    % of fixed size: a flat union.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, int)
    ;       three(float, int)
    ;       four(int, int, int, int)
    ;       five(float)
    ;       six(int)
    ;       seven(int, float, int).

:- func make(int) = fdata.
:- func use(fdata) = int.
""", """
make(N) = D :-
    K = N mod 9,
    ( if K = 0 then D = c0
    else if K = 1 then D = one(N)
    else if K = 2 then D = two(N, -N)
    else if K = 3 then D = three(float(N) / 2.0, N)
    else if K = 4 then D = four(N, 1, 2, 3)
    else if K = 5 then D = five(float(N) * 4.0)
    else if K = 6 then D = six(N * N)
    else if K = 7 then D = seven(N, 0.5, N)
    else D = c1
    ).

use(c0) = 1000.
use(c1) = 2000.
use(one(I)) = I.
use(two(I, J)) = I * 10 + J.
use(three(F, I)) = truncate_to_int(F * 10.0) + I.
use(four(A, B, C, D)) = A * 1000 + B * 100 + C * 10 + D.
use(five(F)) = truncate_to_int(F).
use(six(I)) = I + 7.
use(seven(A, F, B)) = A + truncate_to_int(F * 100.0) + B.
"""),
    "flat-values": (["int", "int8", "uint16", "float"], """
:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, float).

:- func make(int) = fdata.
:- func use(fdata) = int.
""", """
make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    float(N) * 0.25).

use(fdata(C, wrapped(W), I, U, F)) =
    colour_code(C) + W + int8.to_int(I) + uint16.to_int(U) +
    truncate_to_int(F * 100.0).

:- func colour_code(colour) = int.

colour_code(red) = 1.
colour_code(green) = 2.
colour_code(blue) = 3.
"""),
    "equivalence-types": (["int"], """
:- type point
    --->    point(int, int).
:- type distance == int.
:- type fdata == point.

:- func make(int) = fdata.
:- func use(fdata) = distance.
""", """
make(N) = point(N, N * 2).

use(point(X, Y)) = X * X + Y * Y.
"""),
    "field-access": (["int"], """
:- type fdata
    --->    account(
                id :: int,
                balance :: int,
                limit :: int
            ).

:- func make(int) = fdata.
:- func use(fdata) = int.
""", """
make(N) = account(N, N * 10, 50).

use(A0) = A ^ id * 10000 + A ^ balance * 10 + A ^ limit :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ limit := A1 ^ limit - A0 ^ id.
"""),
    "determinism": (["int"], """
:- type fdata
    --->    range(int, int).

:- func make(int) = fdata.
:- func use(fdata) = int.
""", """
make(N) = range(1, N + 2).

    % The first number over 2 in the range, by a semidet search, and how
    % many are even, by a nondet one, aggregated.
use(range(Lo, Hi)) = First * 100 + Evens :-
    ( if first_over(Lo, Hi, 2, F) then First = F else First = 0 ),
    Evens = count_evens(Lo, Hi, 0).

:- pred first_over(int::in, int::in, int::in, int::out) is semidet.

first_over(I, Hi, Limit, F) :-
    I =< Hi,
    ( if I > Limit then
        F = I
    else
        first_over(I + 1, Hi, Limit, F)
    ).

:- pred even_in(int::in, int::in, int::out) is nondet.

even_in(Lo, Hi, X) :-
    in_range(Lo, Hi, X),
    X mod 2 = 0.

:- pred in_range(int::in, int::in, int::out) is nondet.

in_range(Lo, Hi, X) :-
    Lo =< Hi,
    ( X = Lo
    ; in_range(Lo + 1, Hi, X)
    ).

:- func count_evens(int, int, int) = int.

count_evens(Lo, Hi, N0) = N :-
    promise_equivalent_solutions [N] (
        ( if even_in(Lo, Hi, X), X > N0 then
            N = 1 + count_evens(X + 1, Hi, N0)
        else
            N = 0
        )
    ).
"""),
    "unique-modes": (["int", "array"], """
:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = int.
""", """
make(N) = N + 3.

use(Size) = sum(0, squares(0, A0), 0) :-
    array.init(Size, 0, A0).

    % Updates the array in place, as its modes allow.
:- func squares(int, array(int)) = array(int).
:- mode squares(in, array_di) = array_uo is det.

squares(I, A0) = A :-
    ( if I < array.size(A0) then
        array.set(I, I * I, A0, A1),
        A = squares(I + 1, A1)
    else
        A = A0
    ).

:- func sum(int, array(int), int) = int.

sum(I, A, S) =
    ( if I < array.size(A) then sum(I + 1, A, S + A ^ elem(I)) else S ).
"""),
}

CONTEXTS = [
    Context(
        "closure", "Creating higher-order terms", ["int", "list", "string"],
        """
    % A closure that makes and uses a value, and one that captures a value
    % made outside it.
:- func run(int) = string.

run(N) = S :-
    Make = (func(K) = make(K)),
    V = apply(Make, N),
    Use = ((func) = use(V)),
    Both = list.map((func(K) = use(make(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).
""",
        "closures that make, capture and use the values"),
    Context(
        "typeclass-method", "Type classes", ["int", "string"],
        """
:- typeclass runner(T) where [
    func produce(T, int) = fdata,
    func consume(T, fdata) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make(N),
    consume(plain, V) = use(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make(N + K),
    consume(offset(_), V) = "offset " ++ use(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).

:- func run(int) = string.

run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).
""",
        "instance methods that make and use the values"),
    Context(
        "polymorphic", "Predicate and function type declarations",
        ["int", "list", "string"],
        """
    % Polymorphic code that carries the values, in lists, pairs and maybes,
    % compiled once for every type.
:- pred keep(T::in, T::out) is det.
:- pragma no_inline(keep/2).

keep(X, X).

:- type holder(T)
    --->    holder(T, int)
    ;       nothing.

:- func second(list(T), T) = T.
:- pragma no_inline(second/2).

second(Xs, Default) =
    ( if Xs = [_, X | _] then X else Default ).

:- func run(int) = string.

run(N) = S :-
    keep([make(N), make(N + 1)], Vs),
    keep(holder(make(N + 2), N), H),
    ( if H = holder(V, _) then
        Held = use(V)
    else
        Held = "nothing"
    ),
    S = use(second(list.reverse(Vs), make(0))) ++ " | " ++ Held.
""",
        "polymorphic procedures that carry the values"),
    Context(
        "nondet", "Determinism categories",
        ["int", "list", "string", "solutions"],
        """
    % Values made in a nondet goal, each solution's in turn.
:- pred made(int::in, string::out) is nondet.

made(N, S) :-
    list.member(K, [N, N + 1, N + 2]),
    K mod 2 = 1,
    V = make(K),
    S = use(V).

:- func run(int) = string.

run(N) = S :-
    solutions(made(N), Ss),
    S = string.join_list(" | ", Ss).
""",
        "a nondet goal that makes the values"),
    Context(
        "try", "Exception handling",
        ["int", "string", "exception", "univ"],
        """
    % A value thrown as an exception, caught and used.
:- pred thrower(int::in, string::out) is det.

thrower(N, S) :-
    V = make(N),
    ( if N mod 2 = 0 then
        throw(V)
    else
        S = "returned " ++ use(V)
    ).

:- func run(int) = string.

run(N) = S :-
    promise_equivalent_solutions [R] (
        try(thrower(N), R)
    ),
    (
        R = succeeded(S)
    ;
        R = exception(E),
        ( if univ_to_type(E, V) then
            S = "caught " ++ use(V)
        else
            S = "caught something else"
        )
    ).
""",
        "values thrown as exceptions and caught"),
    Context(
        "another-module", "Modules", ["int", "list", "string"],
        """
    % The type, make and use are another module's.
:- func run(int) = string.

run(N) = S :-
    Vs = list.map(make, [N, N + 1]),
    S = string.join_list(" | ", list.map(use, Vs)).
""",
        "the feature in another module"),
]


EXTENSION_CONTEXTS = [
    Context(
        "do-loop", "Loops", ["int", "list", "string"],
        """
    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.
:- func run(int) = string.

run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).
""",
        "a do loop that makes, gathers and carries the values",
        flags=["-fdo-loops"],
        plain="""
:- func run(int) = string.

run(N) = string.join_list(" | ",
    list.map((func(K) = use(make(K))), [N, N + 1])) ++ " | last " ++
    use(make(N + 20)).
"""),
    Context(
        "parallel-do", "Loops / Under a parallel directive",
        ["int", "list", "string", "array"],
        """
    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make(N + I),
        !A ^ unsafe_elem(I) := use(V)
    ).

:- func run(int) = string.

run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).
""",
        "a parallel loop's iterations, which make and use the values",
        flags=["-fdo-loops", "-fopenmp"],
        plain="""
:- func run(int) = string.

run(N) = string.join_list(" | ",
    list.map((func(I) = use(make(N + I))), 0 .. 3)).
"""),
    Context(
        "host-process", "Processes and mailboxes",
        ["int", "list", "string", "process"],
        """
    % Another process makes and uses the values, in its own heap, and
    % sends the text back.
:- pred worker(mailbox(string)::in, mailbox(int)::in, io::di, io::uo)
    is det.

worker(Replies, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(K),
        ( if K < 0 then
            true
        else
            send(Replies, use(make(K)), !IO),
            worker(Replies, Self, !IO)
        )
    ;
        R = timeout
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    new_mailbox(Replies, !IO),
    spawn(worker(Replies), Worker, !IO),
    send(Worker, N, !IO),
    send(Worker, N + 1, !IO),
    send(Worker, -1, !IO),
    reply(Replies, S1, !IO),
    reply(Replies, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred reply(mailbox(string)::in, string::out, io::di, io::uo) is det.

reply(Replies, S, !IO) :-
    receive(Replies, any, infinity, R, !IO),
    ( R = message(S) ; R = timeout, S = "timeout" ).
""",
        "another process, which makes and uses the values",
        plain="""
:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, use(make(N)) ++ " | " ++ use(make(N + 1)), !IO).
""", io=True),
    Context(
        "message", "Processes and mailboxes",
        ["int", "list", "string", "process"],
        """
    % The values as messages: copied into another process's heap, and back.
:- pred echo(int::in, mailbox(fdata)::in, mailbox(fdata)::in,
    io::di, io::uo) is det.

echo(Left, Back, Self, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Self, any, infinity, R, !IO),
        (
            R = message(V),
            send(Back, V, !IO),
            echo(Left - 1, Back, Self, !IO)
        ;
            R = timeout
        )
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    new_mailbox(Back, !IO),
    spawn(echo(2, Back), Echo, !IO),
    send(Echo, make(N), !IO),
    send(Echo, make(N + 1), !IO),
    back(Back, S1, !IO),
    back(Back, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred back(mailbox(fdata)::in, string::out, io::di, io::uo) is det.

back(Back, S, !IO) :-
    receive(Back, any, infinity, R, !IO),
    ( R = message(V), S = use(V) ; R = timeout, S = "timeout" ).
""",
        "messages sent to another process and back",
        plain="""
:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, use(make(N)) ++ " | " ++ use(make(N + 1)), !IO).
""", io=True),
    Context(
        "gpu-process", "GPU processes",
        ["int", "list", "string", "process"],
        """
    % A process kind on gpu, which makes and uses the values where it runs:
    % at the host, or on the GPU when the program's argument is gpu
    % (run-*-gpu.test).
:- type request
    --->    request(int, mailbox(int))
    ;       stop.

:- process worker receives request on gpu.

worker(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(request(K, Reply)),
        send(Reply, use(make(K)), !IO),
        worker(Self, !IO)
    ;
        R = message(stop)
    ;
        R = timeout
    ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, S, !IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Replies, !IO),
    spawn(Place, worker, Worker, !IO),
    send(Worker, request(N, Replies), !IO),
    send(Worker, request(N + 1, Replies), !IO),
    send(Worker, stop, !IO),
    reply(Replies, S1, !IO),
    reply(Replies, S2, !IO),
    S = S1 ++ " | " ++ S2.

:- pred reply(mailbox(int)::in, string::out, io::di, io::uo) is det.

reply(Replies, S, !IO) :-
    receive(Replies, any, infinity, R, !IO),
    ( R = message(K), S = string.int_to_string(K)
    ; R = timeout, S = "timeout"
    ).
""",
        "a GPU process, which makes and uses the values",
        flags=["-fprocess-declarations"],
        plain="""
:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).
""", io=True, gpu=True),
]

CONTEXTS += EXTENSION_CONTEXTS


def mercury_name(*parts):
    return "_".join(p.replace("-", "_") for p in parts)


def imports_line(names):
    return ":- import_module " + ", ".join(sorted(names)) + "."


def test_text(feature, context):
    """The test's text, and its other module's, if it has one."""
    module = mercury_name(context.name, feature.name)
    header = [
        "% MANUAL: " + feature.manual,
    ]
    if context.plain:
        header.append("% DESIGN: " + context.manual)
    elif context.manual != feature.manual:
        header.append("% MANUAL: " + context.manual)
    header += textwrap.wrap(
        "Interaction: %s, in %s." % (feature.summary, context.summary),
        width=76, initial_indent="% ", subsequent_indent="% ")
    header += [
        "% Written by utils/conformance/generate-interactions.py; change",
        "% the generator rather than this file.",
    ]
    main_imports = {"io", "int", "list", "string"} | context.imports
    if context.flags:
        header.append("% MERKC-FLAGS: " + " ".join(context.flags))
    translation = None
    if context.plain:
        translation = module + "_translation"
        header.append("%% TRANSLATION: Inputs/%s.m" % translation)
    lib = None
    if context.name == "another-module":
        library = module + "_lib"
        header.append("%% MODULES: Inputs/%s.m" % library)
        main_imports.add(library)
        lib_text = "\n".join([
            "%% The feature of run-%s.m, as another module." % feature.name,
            ":- module %s." % library,
            ":- interface.",
        ] + ([imports_line(feature.interface_imports)]
             if feature.interface_imports else []) + [
            "",
            feature.interface,
            "",
            ":- implementation.",
        ] + ([imports_line(feature.imports - feature.interface_imports)]
             if feature.imports - feature.interface_imports else []) + [
            "",
            feature.implementation,
            ""])
        lib = (library, lib_text)
        body_feature = ""
    else:
        main_imports |= feature.imports
        body_feature = "\n".join([feature.interface, "",
                                  feature.implementation, ""])
    text = "\n".join(header + program(module, main_imports, body_feature,
                                      context.code, context.io))
    others = [lib] if lib else []
    if translation:
        # mmc has no process module: the translation needs none.
        others.append((translation, "\n".join(
            ["%% The translation of run-%s.m in %s for mmc: the same"
             % (feature.name, context.name),
             "% feature in plain Mercury, which must print the same."] +
            program(translation, main_imports - {"process"}, body_feature,
                    context.plain, context.io))))
    return text, others


def gpu_test(feature, context):
    """The test that runs a GPU context's test on the GPU."""
    test = "run-%s.m" % feature.name
    flags = " ".join(context.flags)
    return "\n".join([
        "%s's GPU process on the machine's GPU, which prints what it" % test,
        "prints at the host, the output the oracle checked. Written by",
        "utils/conformance/generate-interactions.py.",
        "REQUIRES: amdgpu-gpu",
        "",
        "RUN: merkc -O2 %s --offload-arch=%%offload-arch %%libomp-flags "
        "%%S/%s -o %%t2" % (flags, test),
        "RUN: %%gpu-env timeout -k 1 60 %%t2 gpu 2>&1 | FileCheck %%S/%s "
        "--check-prefix=OUTPUT --match-full-lines --strict-whitespace" % test,
        ""])


def program(module, imports, feature, code, io):
    """A test's module, from the feature's code and the context's."""
    lines = [
        ":- module %s." % module,
        ":- interface.",
        ":- import_module io.",
        ":- pred main(io::di, io::uo) is det.",
        ":- implementation.",
        imports_line(imports - {"io"}) if imports - {"io"} else "",
        "",
    ]
    if feature:
        lines += [feature]
    lines += [code, "",
              "main(!IO) :-",
              "    list.foldl(show, 1 .. 4, !IO).",
              "",
              ":- pred show(int::in, io::di, io::uo) is det.",
              ""]
    if io:
        lines += ["show(N, !IO) :-",
                  "    run(N, S, !IO),",
                  '    io.format("%d: %s\\n", [i(N), s(S)], !IO).',
                  ""]
    else:
        lines += ["show(N, !IO) :-",
                  '    io.format("%d: %s\\n", [i(N), s(run(N))], !IO).',
                  ""]
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    changed = []
    for context in CONTEXTS:
        directory = os.path.join(args.out, context.name)
        for feature in FEATURES:
            if context.gpu:
                if feature.name not in GPU_VARIANTS:
                    continue
                imports, interface, implementation = GPU_VARIANTS[
                    feature.name]
                feature = Feature(feature.name, feature.manual, imports,
                                  interface, implementation, feature.summary)
            text, others = test_text(feature, context)
            files = [(os.path.join(directory, "run-%s.m" % feature.name),
                      text)]
            for name, other in others:
                files.append((os.path.join(directory, "Inputs",
                                           name + ".m"), other))
            if context.gpu:
                files.append((os.path.join(
                    directory, "run-%s-gpu.test" % feature.name),
                    gpu_test(feature, context)))
            for path, content in files:
                old = None
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        old = f.read()
                # The oracle adds its lines to the test's header; compare
                # what this writes, the header's other lines and the body.
                if old is not None and strip_generated(old) == content:
                    continue
                changed.append(path)
                if not args.check:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
    for path in changed:
        print(("would change: " if args.check else "wrote: ") +
              os.path.relpath(path))
    sys.exit(1 if args.check and changed else 0)


def strip_generated(text):
    """A test as this writes it, without the lines the oracle adds."""
    generated = ("% RUN:", "% OUTPUT", "% ERRORS", "% NYI", "% MMC",
                 "% STATUS", "% XFAIL:", "% REQUIRES:")
    lines = text.split("\n")
    out = []
    in_header = True
    for line in lines:
        if in_header and not line.startswith("%"):
            in_header = False
            # The oracle's `%' line between the author's and its own.
            while out and out[-1] == "%":
                out.pop()
        if in_header and line.startswith(generated):
            continue
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    main()
