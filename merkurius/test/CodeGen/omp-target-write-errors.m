% The walker is in the host's runtime, so writing a value by its type_info is an
% error on a device, at the write.
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
% CHECK: error: 'merk.write_word' op writes a value by its type_info, which cannot be done on an OpenMP device yet
