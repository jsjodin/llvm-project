% On the device the library's procedures are copied in from its merk
% bytecode, but not yet compiled with their foreign code, so a target region
% cannot call one with a foreign_proc, such as io's beyond the output that has
% a device version (omp-target-write.m): the error is at the foreign code,
% with a note at the region's call (docs/INPROGRESS.md, Library code on the
% device). On the host it runs.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_device_call.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    %$omp target
    io.write_float(1.5, !IO).
% CHECK: io.m:{{[0-9]+}}:{{[0-9]+}}: error: cannot call io.write_float_3_0 on an OpenMP device yet: its foreign code is not compiled for the device
% CHECK-NEXT: omp-target-device-call.m:[[@LINE-2]]:5: note: called by omp_target_device_call.main_2_0 here: io.write_float_3_0

% OUTPUT:1.5
