% The device clause on target and on target data (omp-target-device-loop.m
% has it on a combined target loop): a number, or an integer variable bound before the directive. As Flang's
% (flang/test/Lower/OpenMP/target-spmd.f90), it is the operand device of the
% operation, evaluated on the host. Without a GPU, or without -fopenmp, the
% program gives the same output.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=HOST
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_device.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- pragma no_inline(number/1).
:- pred number(int::out) is det.
number(0).

main(!IO) :-
    number(D),
    One = 1,
    % MERK: %[[ZERO:.*]] = arith.constant 0 : i64
    % MERK: merk.foreign_call @merk_check_region_units(%[[ZERO]]) : (i64) -> ()
    % MERK: omp.target {{.*}}device(%[[ZERO]] : i64)
    %$omp target device(0)
    Two = One + 1,
    % MERK: omp.target {{.*}}device(%{{.*}} : i64)
    %$omp target device(D)
    Three = Two + 1,
    A0 = array.init(2, 0),
    % MERK: omp.target_data device(%{{.*}} : i64)
    %$omp target data map(to: A0) device(D)
    up(A0, A),
    io.print_line({D, Two, Three, A ^ elem(0)}, !IO).

    % HOST: call i32 @__tgt_target_kernel(ptr @{{.*}}, i64 0,

:- pred up(array(int)::array_di, array(int)::array_uo) is det.
up(!A) :-
    !A ^ elem(0) := 7.

% OUTPUT:{0, 2, 3, 7}
