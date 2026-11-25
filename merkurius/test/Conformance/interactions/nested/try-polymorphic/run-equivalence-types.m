% MANUAL: Equivalence types
% MANUAL: Exception handling
% MANUAL: Predicate and function type declarations
% Interaction: a type defined as another, generic one, in polymorphic
% procedures that carry the values, in values thrown as exceptions and
% caught.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_try_polymorphic_equivalence_types_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_try_polymorphic_equivalence_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_try_polymorphic_equivalence_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: returned 1: n1 / 1: n1 | 3: n1 n2 n3
% OUTPUT-NEXT:2: caught 2: n1 n2 / 2: n1 n2 | 4: n1 n2 n3 n4
% OUTPUT-NEXT:3: returned 3: n1 n2 n3 / 3: n1 n2 n3 | 5: n1 n2 n3 n4 n5
% OUTPUT-NEXT:4: caught 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 6: n1 n2 n3 n4 n5 n6
% OUTPUT-NOT:{{.}}
:- module nested_try_polymorphic_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, nested_try_polymorphic_equivalence_types_lib, string, univ.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % A value thrown as an exception, caught and used.
:- pred thrower(int::in, string::out) is det.

thrower(N, S) :-
    V = make_nested(N),
    ( if N mod 2 = 0 then
        throw(V)
    else
        S = "returned " ++ use_nested(V)
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
            S = "caught " ++ use_nested(V)
        else
            S = "caught something else"
        )
    ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
