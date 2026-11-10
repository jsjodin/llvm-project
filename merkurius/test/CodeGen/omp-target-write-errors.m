% io.write is a foreign_proc on the walker in the host's runtime, so it cannot be
% called on a device: the device's compile copies it from the library, and
% does not compile its foreign code.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -fdo-loops -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s
% REQUIRES: amdgpu-registered-target

:- module omp_target_write_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

main(!IO) :-
    %$omp target
    io.write(1, !IO).
% CHECK: io.m:{{[0-9]+}}:{{[0-9]+}}: error: cannot call io.write_3_0 on an OpenMP device yet: its foreign code calls merk_write, which is not on the device
% CHECK-NEXT: omp-target-write-errors.m:{{[0-9]+}}:5: note: called by omp_target_write_errors.main_2_0 here: io.write_3_0
