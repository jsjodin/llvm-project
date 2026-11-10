% A target region in the module's last procedure, which merk.entry names
% beside it: omp-function-filter, which removes the entry from a device's
% module as a use of a host procedure, erased it while walking the module, as
% the operation it would visit next, and crashed.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -O0 -fopenmp %libomp-flags %s -o %t0 && %libomp-env %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t2 && %libomp-env %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_last_proc.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    %$omp target
    N = 41,
    io.write_int(N + 1, !IO),
    io.nl(!IO).

% DEVICE: define {{.*}}@__omp_offloading_{{.*}}omp_target_last_proc.main_2_0{{.*}}(
% DEVICE-NOT: merk_main

% OUTPUT:42
% OUTPUT-NOT:{{.}}
