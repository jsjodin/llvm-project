% On the device only this module's procedures are compiled so far, so a
% target region cannot call the library's, such as io's beyond the output that
% has a device version (omp-target-write.m): the error is at the call. On the
% host it runs.
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
% CHECK: omp-target-device-call.m:[[@LINE-1]]:5: error: cannot call io.write_float_3_0 on an OpenMP device yet: it is not compiled for the device

% OUTPUT:1.5
