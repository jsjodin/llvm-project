% A comparison at a known type needs no descriptor, so it compiles for a
% device, and the kernel's result is the host's.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --match-full-lines
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --match-full-lines
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --match-full-lines
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --match-full-lines
% REQUIRES: amdgpu-registered-target

:- module omp_target_compare.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- type colour ---> red ; green ; blue.

:- pred fill_it(int::in, array(int)::array_di, array(int)::array_uo) is det.
fill_it(N, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        compare(R, I, 1),
        (
            R = (<),
            compare(C, red, blue),
            ( C = (<), V = 1 ; C = (=), V = 4 ; C = (>), V = 5 )
        ;
            R = (=),
            V = 2
        ;
            R = (>),
            V = 3
        ),
        !A ^ unsafe_elem(I) := V
    ).

main(!IO) :-
    array.init(3, 0, A0),
    fill_it(3, A0, A),
    % CHECK:1
    io.write_int(array.lookup(A, 0), !IO),
    io.nl(!IO),
    % CHECK-NEXT:2
    io.write_int(array.lookup(A, 1), !IO),
    io.nl(!IO),
    % CHECK-NEXT:3
    io.write_int(array.lookup(A, 2), !IO),
    io.nl(!IO).
