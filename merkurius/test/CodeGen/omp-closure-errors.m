% A closure cannot be made or called on a device yet: the device image is
% another binary, and a device does no run-time dispatch (design document,
% Lowering, "Closures"). Higher-order specialization will make such calls
% direct.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
% REQUIRES: amdgpu-registered-target
%
% CHECK-DAG: omp-closure-errors.m:25:10: error: 'merk.closure' op makes a closure, which cannot be done on an OpenMP device yet
% CHECK-DAG: omp-closure-errors.m:26:9: error: 'merk.call_closure' op calls a closure, which cannot be done on an OpenMP device yet

:- module omp_closure_errors.
:- interface.
:- import_module array.
:- pred passes(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- implementation.
:- import_module int.
passes(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := next(I)
    ).
:- func next(int) = int.
next(X) = Y :-
    F = (func(Z) = Z + 1),
    Y = F(X).
