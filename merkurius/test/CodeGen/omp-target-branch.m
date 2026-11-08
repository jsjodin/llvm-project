% A target region in a branch of an if-then-else, given an array that the
% branch found. In the device's compile, omp-host-op-filter keeps only the
% region of the host procedure, and replaces the values its map entries
% use by placeholders; it kept the arguments of every block of the
% procedure, but only the entry block's go with the new procedure, so the
% compile stopped on an assertion when a map entry used an argument of the
% branch's block (this tree's change to the filter).
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_branch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, maybe.

:- pred find(array(int)::in, maybe(array(int))::out) is det.
find(A, yes(A)).

:- func sum(array(int), int, int) = int.
sum(A, I, S) = ( if I >= size(A) then S else sum(A, I + 1, S + A ^ elem(I)) ).

main(!IO) :-
    A0 = array([1, 2, 3, 4]),
    find(A0, Found),
    ( if Found = yes(A) then
        %$omp target
        S = sum(A, 0, 0),
        io.write_int(S, !IO)
    else
        io.write_string("none", !IO)
    ),
    io.nl(!IO).
% DEVICE: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_omp_target_branch.main_2_0_l{{[0-9]+}}(

% OUTPUT:10
