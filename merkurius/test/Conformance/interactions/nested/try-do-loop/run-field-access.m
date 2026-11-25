% MANUAL: Field access functions
% MANUAL: Exception handling
% DESIGN: Loops
% Interaction: field access and update functions, in a do loop that makes,
% gathers and carries the values, in values thrown as exceptions and caught.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_try_do_loop_field_access_translation.m
% MODULES: Inputs/nested_try_do_loop_field_access_lib.m
% FILES: Inputs/nested_try_do_loop_field_access_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_try_do_loop_field_access_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_try_do_loop_field_access_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: returned owner1 15 10 / owner1 15 10 | owner2 25 20 | last owner21 215 210
% OUTPUT-NEXT:2: caught owner2 25 20 / owner2 25 20 | owner3 35 30 | last owner22 225 220
% OUTPUT-NEXT:3: returned owner3 35 30 / owner3 35 30 | owner4 45 40 | last owner23 235 230
% OUTPUT-NEXT:4: caught owner4 45 40 / owner4 45 40 | owner5 55 50 | last owner24 245 240
% OUTPUT-NOT:{{.}}
:- module nested_try_do_loop_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, nested_try_do_loop_field_access_lib, string, univ.

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
