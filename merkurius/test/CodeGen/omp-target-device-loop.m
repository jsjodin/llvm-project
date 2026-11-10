% The device clause on a combined target loop, on the target leaf.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_device_loop.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred sum(int::in, int::in, int::out) is det.
sum(D, Hi, Sum) :-
    %$omp target teams distribute parallel do reduction((+) : Sum) device(D)
    ( for(I, 1, Hi), fromto(0, S0, S1, Sum) do
        S1 = S0 + I
    ).
% MERK: omp.target {{.*}}device(%arg0 : i64)

main(!IO) :-
    sum(0, 100, A),
    io.write_int(A, !IO), io.nl(!IO).

% OUTPUT:5050
