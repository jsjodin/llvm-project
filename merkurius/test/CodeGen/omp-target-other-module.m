% A region that calls a procedure of another module of the program, which is
% not the library's, is an error in the device's compile: only the library's
% modules are compiled for a device yet (design document, Library code on
% the device).
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -I %S/Inputs -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=note:
% RUN: merkc -fc1 -fopenmp -I %S/Inputs -emit-merk %s -o /dev/null
% REQUIRES: amdgpu-registered-target

:- module omp_target_other_module_use.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, omp_target_other_module.

main(!IO) :-
    %$omp target
    N = triple(14),
    io.write_int(N, !IO),
    io.nl(!IO).

% CHECK: omp-target-other-module.m:[[#@LINE-4]]:9: error: cannot call omp_target_other_module.triple_f1_0 on an OpenMP device yet: it is not compiled for the device
