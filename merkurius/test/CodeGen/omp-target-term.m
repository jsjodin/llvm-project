% A target region may make a term. On a device, which has no heap of ours,
% a cell comes from the device's runtime, merk_device_alloc, which takes it
% from the OpenMP device runtime's malloc (runtime/device); a ground term is
% a static cell, as on the host. On the host the region allocates from the
% heap.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_term.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

:- func make(int) = list(int).
make(N) = [N, N + 1].
% Two cells of [|], of two words each.
% CHECK-LABEL: define internal ptr @omp_target_term.make_f1_0(
% CHECK-COUNT-2: call ptr @merk_device_alloc(i64 16)
% CHECK-NOT: merk_alloc_slow
% CHECK: declare ptr @merk_device_alloc(i64)

main(!IO) :-
    N = 3,
    %$omp target
    S = sum(make(N), 0) + sum([1, 2], 0),
    io.write_int(S, !IO),
    io.nl(!IO).
% OUTPUT:10
