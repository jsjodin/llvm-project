% DESIGN: Sized floating point types
% float32s in target regions: a loop fills an array of float32s, packed 4
% bytes to an element, on the device; reductions with + and max, and an
% atomic update, are of float32s, as of floats; and math_float32's sqrt
% runs on the device, as sqrtf. The values are sums of quarters, exact in
% float32 in any order, so the threads' order cannot show.
% run-float32-target-gpu.test runs it on the GPU.
% MERKC-FLAGS: -fopenmp -fdo-loops
% TRANSLATION: Inputs/float32_target_translation.c
%
% REQUIRES: libomp
% RUN: merkc -O0 -fopenmp -fdo-loops %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:5.25
% OUTPUT-NEXT:250000.0
% OUTPUT-NEXT:499.75
% OUTPUT-NEXT:array([62.5f32, 62.5f32, 62.5f32, 62.5f32])
% OUTPUT-NOT:{{.}}
:- module run_float32_target.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, float32, int, math_float32.

:- pred fill(int::in, array(float32)::array_di, array(float32)::array_uo)
    is det.

fill(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        X = float32.cast_from_int(I),
        !A ^ unsafe_elem(I) := sqrt(X * X) * 0.5f32 + 0.25f32
    ).

:- func total(int, array(float32)) = float32.

total(N, A) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0f32, S0, S1, Sum) do
        S1 = S0 + A ^ unsafe_elem(I)
    ).

:- func largest(int, array(float32)) = float32.

largest(N, A) = Max :-
    %$omp target teams distribute parallel do reduction(max : Max)
    ( for(I, 0, N - 1), fromto(-1.0f32, M0, M1, Max) do
        M1 = float32.max(M0, A ^ unsafe_elem(I))
    ).

:- pred bins(int::in, array(float32)::array_di, array(float32)::array_uo)
    is det.

bins(N, !B) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        add_quarter(I mod 4, !B)
    ).

:- pred add_quarter(int::in, array(float32)::array_di,
    array(float32)::array_uo) is det.

add_quarter(K, !B) :-
    %$omp atomic update
    !B ^ elem(K) := !.B ^ elem(K) + 0.25f32.

main(!IO) :-
    N = 1000,
    fill(N, array.init(N, 0.0f32), A),
    io.write_float32(A ^ elem(10), !IO), io.nl(!IO),
    io.write_float32(total(N, A), !IO), io.nl(!IO),
    io.write_float32(largest(N, A), !IO), io.nl(!IO),
    bins(N, array.init(4, 0.0f32), B),
    io.write(B, !IO), io.nl(!IO).
