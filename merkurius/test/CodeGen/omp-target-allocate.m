% A target region allocates more than the OpenMP device runtime's fallback
% allocator has, a bump allocator of 1 MiB that traps when it is full: the
% list of 200000 cells of 16 bytes is 3.2 MB. On a GPU, merk_device_alloc
% is the OpenMP device runtime's malloc, which is LLVM libc's GPU allocator
% when the OpenMP device runtime is built with libc for the GPU, as
% build_merk.sh builds it (omp-target-allocate-gpu.test).
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_allocate.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func build(int, list(int)) = list(int).
build(N, L) = ( if N = 0 then L else build(N - 1, [N | L]) ).

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

main(!IO) :-
    N = 200000,
    %$omp target
    S = sum(build(N, []), 0),
    io.write_int(S, !IO),
    io.nl(!IO).

% OUTPUT:20000100000
