% MANUAL: Calling higher-order terms
% DESIGN: Loops / Under a parallel directive
% DESIGN: Loops
% Interaction: closures stored in a term and called, in a do loop that
% makes, gathers and carries the values, in a parallel loop's iterations,
% which make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_parallel_do_do_loop_closures_as_data_translation.m
% MODULES: Inputs/nested_parallel_do_do_loop_closures_as_data_lib.m
% FILES: Inputs/nested_parallel_do_do_loop_closures_as_data_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_do_loop_closures_as_data_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_do_loop_closures_as_data_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_do_loop_closures_as_data_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_do_loop_closures_as_data_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 11 3/1 / 11 3/1 | 21 5/2 | last 211 43/21 | 21 5/2 / 21 5/2 | 31 7/3 | last 221 45/22 | 31 7/3 / 31 7/3 | 41 9/4 | last 231 47/23 | 41 9/4 / 41 9/4 | 51 11/5 | last 241 49/24
% OUTPUT-NEXT:2: 21 5/2 / 21 5/2 | 31 7/3 | last 221 45/22 | 31 7/3 / 31 7/3 | 41 9/4 | last 231 47/23 | 41 9/4 / 41 9/4 | 51 11/5 | last 241 49/24 | 51 11/5 / 51 11/5 | 61 13/6 | last 251 51/25
% OUTPUT-NEXT:3: 31 7/3 / 31 7/3 | 41 9/4 | last 231 47/23 | 41 9/4 / 41 9/4 | 51 11/5 | last 241 49/24 | 51 11/5 / 51 11/5 | 61 13/6 | last 251 51/25 | 61 13/6 / 61 13/6 | 71 15/7 | last 261 53/26
% OUTPUT-NEXT:4: 41 9/4 / 41 9/4 | 51 11/5 | last 241 49/24 | 51 11/5 / 51 11/5 | 61 13/6 | last 251 51/25 | 61 13/6 / 61 13/6 | 71 15/7 | last 261 53/26 | 71 15/7 / 71 15/7 | 81 17/8 | last 271 55/27
% OUTPUT-NOT:{{.}}
:- module nested_parallel_do_do_loop_closures_as_data.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, nested_parallel_do_do_loop_closures_as_data_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use_nested makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make_nested(N + I),
        !A ^ unsafe_elem(I) := use_nested(V)
    ).

:- func run(int) = string.

run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
