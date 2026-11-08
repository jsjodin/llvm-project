% A procedure that a target region calls may allocate an array, which the
% region then gives: on the host it is the new array, and on a GPU (the -gpu
% test) the device copies it to the host before the region ends, since its
% arena ends with the region. The output is mmc's.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s -o /dev/null
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
