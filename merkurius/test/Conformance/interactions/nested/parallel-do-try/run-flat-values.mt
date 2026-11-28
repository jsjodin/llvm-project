% MANUAL: Discriminated unions
% DESIGN: Loops / Under a parallel directive
% MANUAL: Exception handling
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in values
% thrown as exceptions and caught, in a parallel loop's iterations, which
% make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_parallel_do_try_flat_values_translation.mt
% MODULES: Inputs/nested_parallel_do_try_flat_values_lib.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_parallel_do_try_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_parallel_do_try_flat_values_float.m %t.float.d/nested_parallel_do_try_flat_values_lib.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float.d %t.float.d/nested_parallel_do_try_flat_values_float.m %t.float.d/nested_parallel_do_try_flat_values_lib.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_parallel_do_try_flat_values_float.m %t.float.d/nested_parallel_do_try_flat_values_lib.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_parallel_do_try_flat_values_float.m %t.float.d/nested_parallel_do_try_flat_values_lib.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 / returned green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 / caught blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 / caught blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / caught red 600 -6 6000 1.5
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / caught red 600 -6 6000 1.5 | green 700 -7 7000 1.75 / returned green 700 -7 7000 1.75
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_parallel_do_try_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_parallel_do_try_flat_values_float32.m %t.float32.d/nested_parallel_do_try_flat_values_lib.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %t.float32.d %t.float32.d/nested_parallel_do_try_flat_values_float32.m %t.float32.d/nested_parallel_do_try_flat_values_lib.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_parallel_do_try_flat_values_float32.m %t.float32.d/nested_parallel_do_try_flat_values_lib.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_parallel_do_try_flat_values_float32.m %t.float32.d/nested_parallel_do_try_flat_values_lib.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 / returned green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 / caught blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 / caught blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 / returned red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / caught red 600 -6 6000 1.5
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 / caught green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / returned blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / caught red 600 -6 6000 1.5 | green 700 -7 7000 1.75 / returned green 700 -7 7000 1.75
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_parallel_do_try_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, nested_parallel_do_try_flat_values_lib, string.

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
