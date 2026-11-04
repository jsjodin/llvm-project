% A polymorphic procedure called in a target region: its type_info is the
% address of a static descriptor, on the device too, and nothing there reads
% it yet.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_polymorphic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred id(T::in, T::out) is det.
id(X, X).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo) is det.
fill(N, K, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        id(I * K, V),
        !A ^ unsafe_elem(I) := V
    ).
% CHECK: call i64 @omp_target_polymorphic.id_2_0(ptr @"merk.type_ctor.int/0", i64

main(!IO) :-
    array.init(4, 0, A0),
    fill(4, 3, A0, A),
    io.write_int(array.lookup(A, 3), !IO),
    io.nl(!IO).

% OUTPUT:9
