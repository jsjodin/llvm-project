% A target region may use an array that a procedure only the host runs
% made. The device compile removes that procedure and its call, whose
% result omp-function-filter replaces by ub.poison; lowering gives it the
% type the array lowers to, and omp-host-op-filter makes it an argument of
% the function, so that the kernel reads the array from its own argument,
% not from a poison pointer, which translation could not replace.
% The output is mmc's for the same program, without the directive.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-mlir %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_host_made.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- func make(int) = array(int).
make(N) = array.init(N, 2).

main(!IO) :-
    A = make(3),
    %$omp target
    S = array.lookup(A, 0) + array.lookup(A, 2),
    io.write_int(S, !IO),
    io.nl(!IO).
% The array is the host function's argument that omp-host-op-filter
% made of the ub.poison.
% DEVICE-LABEL: llvm.func @omp_target_host_made.main_2_0(%arg0: !llvm.ptr,
% DEVICE: omp.map.info var_ptr(%arg0 : !llvm.ptr, i64) map_clauses(to) capture(ByRef) name("A")
% DEVICE-NOT: llvm.func {{.*}}@omp_target_host_made.make_f1_0

% LLVM-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_omp_target_host_made.main_2_0_l{{[0-9]+}}(
% LLVM-NOT: ptr poison
% LLVM: ret void

% OUTPUT:4
