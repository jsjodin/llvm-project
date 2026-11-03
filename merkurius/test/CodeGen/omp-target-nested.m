% The directive belongs to the caller: a target region in a procedure that
% a target region calls would run on the device, as reverse offloading
% would, so it is an error, in the host compile and the device's.
%
% RUN: not merkc -fc1 -fopenmp -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_nested.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred inner(int::in, int::out) is det.
inner(X, Y) :-
    %$omp target
    Y = X + 1.
% CHECK: omp-target-nested.m:[[@LINE-2]]:5: error: a target region cannot be in omp_target_nested.inner_2_0, which a target region calls

:- pred outer(int::in, int::out) is det.
outer(X, Z) :-
    %$omp target
    ( inner(X, Y), Z = Y * 2 ).

main(!IO) :-
    outer(1, Z), io.write_int(Z, !IO), io.nl(!IO).
