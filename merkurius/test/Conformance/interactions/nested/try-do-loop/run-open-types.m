% DESIGN: Open terms (Prolog-style variables)
% MANUAL: Exception handling
% DESIGN: Loops
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in a do loop that makes, gathers and carries the values, in
% values thrown as exceptions and caught.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopen-types
% TRANSLATION: Inputs/nested_try_do_loop_open_types_translation.m
% MODULES: Inputs/nested_try_do_loop_open_types_lib.m
% FILES: Inputs/nested_try_do_loop_open_types_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -fopen-types -I %S/Inputs %s %S/Inputs/nested_try_do_loop_open_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopen-types -I %S/Inputs %s %S/Inputs/nested_try_do_loop_open_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: returned 1 10 rest  / 1 10 rest  | 1 2 rest 20 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,210
% OUTPUT-NEXT:2: caught 1 2 rest 20 / 1 2 rest 20 | 1 2 rest 3,30 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,220
% OUTPUT-NEXT:3: returned 1 2 rest 3,30 / 1 2 rest 3,30 | 1 2 rest 3,4,40 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230
% OUTPUT-NEXT:4: caught 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240
% OUTPUT-NOT:{{.}}
:- module nested_try_do_loop_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, nested_try_do_loop_open_types_lib, string, univ.

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
