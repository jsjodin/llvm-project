% Parallel loops in predicates with type class constraints, on the host's
% threads, whose body predicates are given the constraints, and reductions of
% what their methods give; and a target region calling a method of an instance
% known there, directly, which needs no typeclass_info on the device.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && env OMP_NUM_THREADS=4 %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && env OMP_NUM_THREADS=4 %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_typeclass.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- typeclass num(T) where [
    func twice(T) = T,
    func as_int(T) = int,
    func as_float(T) = float
].
:- instance num(int) where [
    twice(X) = 2 * X,
    as_int(X) = X,
    as_float(X) = float(X)
].
:- instance num(float) where [
    twice(X) = 2.0 * X,
    as_int(X) = truncate_to_int(X),
    as_float(X) = X
].

:- pred fill(int::in, T::in, array(T)::array_di, array(T)::array_uo) is det
    <= num(T).
fill(N, V, !A) :-
    %$omp parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := twice(V)
    ).

:- pred total(int::in, array(T)::array_ui, int::out, float::out) is det <= num(T).
total(N, A, S, F) :-
    %$omp parallel do reduction((+) : S, F)
    ( for(I, 0, N - 1), fromto(0, S0, S1, S), fromto(0.0, F0, F1, F) do
        S1 = S0 + as_int(A ^ unsafe_elem(I)),
        F1 = F0 + as_float(A ^ unsafe_elem(I))
    ).

:- pred doubles(int::in, array(int)::array_di, array(int)::array_uo) is det.
doubles(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := twice(I)
    ).
% DEVICE: define {{.*}}@"omp_typeclass.twice_f1_0$omp_typeclass.num(int)"
% DEVICE-NOT: merk.instance.omp_typeclass.num(int)"{{.*}}=

main(!IO) :-
    array.init(100, 0, A0),
    fill(100, 21, A0, A),
    total(100, A, S, F),
    io.write_int(S, !IO), io.nl(!IO),
    io.write_float(F, !IO), io.nl(!IO),
    array.init(4, 0.0, B0),
    fill(4, 0.25, B0, B),
    io.write_float(array.lookup(B, 2), !IO), io.nl(!IO),
    array.init(4, 0, C0),
    doubles(4, C0, C),
    io.write_int(array.lookup(C, 3), !IO), io.nl(!IO).
% OUTPUT: 4200
% OUTPUT-NEXT: 4200.0
% OUTPUT-NEXT: 0.5
% OUTPUT-NEXT: 6
