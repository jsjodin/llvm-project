% io.write is a foreign_proc on the walker in the host's runtime, so it cannot be
% called on a device, as any procedure that is not compiled for the device.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgcn-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
% REQUIRES: amdgpu-registered-target

:- module omp_target_write_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    %$omp target
    io.write(1, !IO).
% CHECK: error: cannot call io.write_3_0 on an OpenMP device yet: it is not compiled for the device
