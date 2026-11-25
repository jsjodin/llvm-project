% MANUAL: Discriminated unions
% MANUAL: Exception handling
% DESIGN: Loops / Under a parallel directive
% Interaction: an enumeration, a notag and a record of numbers, in a
% parallel loop's iterations, which make and use the values, in values
% thrown as exceptions and caught.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_try_parallel_do_flat_values_translation.m
% MODULES: Inputs/nested_try_parallel_do_flat_values_lib.m
% FILES: Inputs/nested_try_parallel_do_flat_values_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_try_parallel_do_flat_values_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_try_parallel_do_flat_values_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_try_parallel_do_flat_values_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_try_parallel_do_flat_values_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: returned green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-NEXT:2: caught blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-NEXT:3: returned red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-NEXT:4: caught green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-NOT:{{.}}
:- module nested_try_parallel_do_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, nested_try_parallel_do_flat_values_lib, string, univ.

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
