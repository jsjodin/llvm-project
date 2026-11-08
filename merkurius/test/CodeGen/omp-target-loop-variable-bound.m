% A loop's bound that is a variable, with no goal of its own, in a target
% region: the host gives it as a host_eval value and the device reads it
% from its map, so the host's region must map it as well, or the host's
% kernel call has fewer arguments than the device's kernel reads, and the
% device runs the loop over garbage bounds (the -gpu test). The output is
% what the sequential program prints.
%
% RUN: merkc -fc1 -fopenmp -fdo-loops -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_loop_variable_bound.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred sum(int::in, int::in, int::out) is det.
sum(Lo, Hi, Sum) :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, Lo, Hi), fromto(0, S0, S1, Sum) do
        S1 = S0 + I
    ).
% MERK-LABEL: merk.proc private @omp_target_loop_variable_bound.sum_3_0(
% MERK-DAG:     name("Lo")
% MERK-DAG:     name("Hi")
% MERK:         omp.target kernel_type(spmd) host_eval(%arg0 -> %[[LO:[a-z0-9]+]], %arg1 -> %[[HI:[a-z0-9]+]], %{{.*}} -> %[[STEP:[a-z0-9]+]] : i64, i64, i64)
% MERK:         omp.loop_nest (%{{.*}}) : i64 = (%[[LO]]) to (%[[HI]]) inclusive step (%[[STEP]]) {

main(!IO) :-
    sum(1, 100, A),
    io.write_int(A, !IO), io.nl(!IO),
    sum(3, 7, B),
    io.write_int(B, !IO), io.nl(!IO).

% OUTPUT:5050
% OUTPUT-NEXT:25
