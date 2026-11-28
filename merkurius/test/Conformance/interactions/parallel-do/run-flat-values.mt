% MANUAL: Discriminated unions
% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in a
% parallel loop's iterations, which make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/parallel_do_flat_values_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float.d/parallel_do_flat_values_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float.d/parallel_do_flat_values_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float.d/parallel_do_flat_values_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float.d/parallel_do_flat_values_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float32.d/parallel_do_flat_values_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float32.d/parallel_do_flat_values_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float32.d/parallel_do_flat_values_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float32.d/parallel_do_flat_values_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT32-NOT:{{.}}
:- module parallel_do_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, array, int, int8, list, string, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag,
    % which Merkurius holds by value.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, @T@).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    @FROM_INT@(N) * 0.25@S@).

use(fdata(C, wrapped(W), I, U, F)) =
    string.string(C) ++ " " ++ string.int_to_string(W) ++ " " ++
    string.int_to_string(int8.to_int(I)) ++ " " ++
    string.int_to_string(uint16.to_int(U)) ++ " " ++
    string.@T@_to_string(F).

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

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
