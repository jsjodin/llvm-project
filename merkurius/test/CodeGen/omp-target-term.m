% A target region may use a ground term, which is a static cell, but not
% yet make one, since the device has no heap: the error is at the
% construction, in the device compile. On the host the region allocates.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_term.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

:- func make(int) = list(int).
make(N) = [N, N + 1].
% CHECK: omp-target-term.m:[[@LINE-1]]:11: error: cannot allocate a term on an OpenMP device yet: the device has no heap

main(!IO) :-
    N = 3,
    %$omp target
    S = sum(make(N), 0) + sum([1, 2], 0),
    io.write_int(S, !IO),
    io.nl(!IO).
% OUTPUT:10
