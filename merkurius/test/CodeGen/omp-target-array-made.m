% A procedure that a target region calls may allocate an array on the
% host, but not yet on the device, which has no runtime to allocate it:
% the error is at the call, in the device compile. On the host the region
% gives the new array.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_array_made.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred new(int::in, array(int)::array_di, array(int)::array_uo) is det.
new(N, A0, A) :-
    ( if N > 2 then
        A = array.init(N, 1)
% CHECK: omp-target-array-made.m:[[@LINE-1]]:13: error: cannot call merk_array_new on an OpenMP device yet: it is not compiled for the device
    else
        array.set(0, 5, A0, A)
    ).

main(!IO) :-
    X0 = array.init(2, 0),
    %$omp target
    new(3, X0, X),
    io.write_int(array.size(X), !IO), io.write_string(" ", !IO),
    io.write_int(X ^ elem(0), !IO), io.nl(!IO).

% OUTPUT:3 1
