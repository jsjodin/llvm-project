% parallel do in a target region: with a reduction and a scalar from outside the
% loop, in the region and in a procedure that the region calls, with a bound read
% from memory. On a GPU (omp-target-parallel-do-gpu.test) the team's threads share
% what the loop uses.
%
% RUN: merkc -fc1 -fdo-loops -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fdo-loops -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %libomp-flags %s -o %t.optimized && %libomp-env %t.optimized | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_parallel_do.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

% MERK-LABEL: merk.proc private @omp_target_parallel_do.sum_of_f1_0
% MERK: merk.foreign_call @merk_device_shared_alloc
% MERK: omp.parallel {
% MERK: llvm.load
% MERK: omp.wsloop reduction
% MERK: merk.foreign_call @merk_device_shared_free

:- func sum_of(array(int)) = int.
sum_of(A) = Sum :-
    Last = array.size(A) - 1,
    %$omp parallel do reduction((+) : Sum)
    ( for(I, 0, Last), fromto(0, S0, S1, Sum) do
        S1 = S0 + A ^ elem(I)
    ).

main(!IO) :-
    N = 100,
    %$omp target
    (
        A = array.init(N, 3),
        InProcedure = sum_of(A),
        %$omp parallel do reduction((+) : InRegion)
        ( for(I, 0, N - 1), fromto(0, S0, S1, InRegion) do
            S1 = S0 + I * (A ^ elem(I))
        )
    ),
    io.write_int(InProcedure, !IO),
    io.nl(!IO),
    io.write_int(InRegion, !IO),
    io.nl(!IO).

% OUTPUT:300
% OUTPUT-NEXT:14850
