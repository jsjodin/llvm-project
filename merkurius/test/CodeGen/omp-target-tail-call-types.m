% A last call in device code, whose calling convention is not the tail one:
% LLVM takes a musttail call only between procedures of the same type, so
% a last call of a procedure of another type is an ordinary call. The
% output is mmc's for the same program, without the directive.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_tail_call_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func build(int, list(int)) = int.
build(N, Xs) = ( if N = 0 then size_of(Xs) else build(N - 1, [N | Xs]) ).
% DEVICE-LABEL: define {{.*}} @omp_target_tail_call_types.build_f2_0(
% DEVICE-NOT: musttail call {{.*}}size_of
% DEVICE: call i64 @omp_target_tail_call_types.size_of_f1_0(
% DEVICE: musttail call i64 @omp_target_tail_call_types.build_f2_0(

:- func size_of(list(int)) = int.
size_of(Xs) = count(Xs, 0).

:- func count(list(int), int) = int.
count([], C) = C.
count([_ | Xs], C) = count(Xs, C + 1).

main(!IO) :-
    %$omp target
    X = build(1000, []),
    io.write_int(X, !IO), io.nl(!IO).

% OUTPUT:1000
