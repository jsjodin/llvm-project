% Every array operation that allocates or fills, in a target region: on a
% GPU (the -gpu test) they are the device runtime's, compiled from
% runtime/array.cpp, and allocate in the device's arena. The region uses
% what it makes, and gives one array, which is copied to the host. The
% output is mmc's for the same program, without the directive.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_array_operations.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- func total(array(int), int, int) = int.
total(A, I, S) =
    ( if I >= array.size(A) then S else total(A, I + 1, S + A ^ elem(I)) ).

main(!IO) :-
    %$omp target
    (
        A0 = array.init(4, 2),
        array.set(1, 5, A0, A1),
        array.copy(A1, B0),
        array.set(0, 9, B0, B),
        array.resize(6, 3, A1, C),
        D0 = array.from_list([1, 2, 3]),
        array.fill(8, D0, D),
        Sum = total(B, 0, 0) + total(C, 0, 0) + total(D, 0, 0)
    ),
% MERK: merk.foreign_call @merk_device_give_array(
% DEVICE-DAG: call {{.*}} @merk_array_new(
% DEVICE-DAG: call {{.*}} @merk_array_copy(
% DEVICE-DAG: call {{.*}} @merk_array_resize(
% DEVICE-DAG: call {{.*}} @merk_array_from_list(
% DEVICE-DAG: call {{.*}} @merk_array_fill(
% DEVICE-DAG: call {{.*}} @merk_device_give_array(
    io.write_int(Sum, !IO), io.nl(!IO),
    io.write_int(array.size(C), !IO), io.nl(!IO),
    io.write_int(total(C, 0, 0), !IO), io.nl(!IO),
    io.write_int(C ^ elem(1), !IO), io.write_string(" ", !IO),
    io.write_int(C ^ elem(5), !IO), io.nl(!IO).

% OUTPUT:59
% OUTPUT-NEXT:6
% OUTPUT-NEXT:17
% OUTPUT-NEXT:5 3
