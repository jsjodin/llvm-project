% What needs a typeclass_info cannot be on a device yet, which has no tables and
% nothing maps one: making one in a procedure a target region calls, and
% dispatching through one; and a target region in a procedure that is given one.
%
% RUN: split-file %s %t
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %t/passes.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
% RUN: not merkc -fc1 -fopenmp -fdo-loops -emit-merk %t/within.m -o /dev/null 2>&1 | FileCheck %s --check-prefix=WITHIN
% REQUIRES: amdgpu-registered-target
%
% PASSES: passes.m:10:10: error: 'merk.call_method' op dispatches on a type class, which cannot be done on an OpenMP device yet
% PASSES: passes.m:14:32: error: 'merk.typeclass_info' op dispatches on a type class, which cannot be done on an OpenMP device yet
% WITHIN: within.m:9:5: error: not yet implemented: code generation for target regions in polymorphic procedures

%--- passes.m
:- module passes.
:- interface.
:- import_module array.
:- pred passes(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- implementation.
:- import_module int.
:- typeclass num(T) where [ func twice(T) = T ].
:- instance num(int) where [ twice(X) = 2 * X ].
:- func use(T) = T <= num(T).
use(X) = twice(X).
passes(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := use(I)
    ).

%--- within.m
:- module within.
:- interface.
:- import_module array.
:- typeclass num(T) where [ func twice(T) = T ].
:- pred within(int::in, T::in, array(int)::array_di, array(int)::array_uo) is det <= num(T).
:- implementation.
:- import_module int.
within(N, _, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := I
    ).
