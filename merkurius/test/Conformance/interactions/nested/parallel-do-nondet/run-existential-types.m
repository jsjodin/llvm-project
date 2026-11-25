% MANUAL: Existentially typed data types
% DESIGN: Loops / Under a parallel directive
% MANUAL: Determinism categories
% Interaction: an existentially typed constructor and its dictionary, in a
% nondet goal that makes the values, in a parallel loop's iterations, which
% make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_parallel_do_nondet_existential_types_translation.m
% MODULES: Inputs/nested_parallel_do_nondet_existential_types_lib.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_nondet_existential_types_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_nondet_existential_types_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_nondet_existential_types_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_nondet_existential_types_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: string z / string z | string zzz | int 14 / string zzz | string zzz / string zzz | string zzzzz | int 28 / string zzzzz
% OUTPUT-NEXT:2: int 14 / string zzz | string zzz / string zzz | string zzzzz | int 28 / string zzzzz | string zzzzz / string zzzzz | string zzzzzzz
% OUTPUT-NEXT:3: string zzz / string zzz | string zzzzz | int 28 / string zzzzz | string zzzzz / string zzzzz | string zzzzzzz | int 42 / string zzzzzzz
% OUTPUT-NEXT:4: int 28 / string zzzzz | string zzzzz / string zzzzz | string zzzzzzz | int 42 / string zzzzzzz | string zzzzzzz / string zzzzzzz | string zzzzzzzzz
% OUTPUT-NOT:{{.}}
:- module nested_parallel_do_nondet_existential_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, nested_parallel_do_nondet_existential_types_lib, string.

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
