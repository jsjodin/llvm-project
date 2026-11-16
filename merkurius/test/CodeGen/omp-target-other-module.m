% A region that calls a procedure of another module of the program, which is
% not the library's: the device's compile copies it from that module's merk
% bytecode, which the driver writes when it is given both sources
% (process-other-module.m; design document, Library code on the device).
% Given no directory of the program's modules (-device-module-dir), as when
% each source is compiled on its own, the call is an error at the device's
% compile.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -I %S/Inputs -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=note:
% RUN: merkc -fc1 -fopenmp -I %S/Inputs -emit-merk %s -o /dev/null
%
% With the other module's bytecode, the region's code has a copy of triple,
% private to the device's module.
% RUN: rm -rf %t.dir && mkdir %t.dir
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -fno-copy-device-procedures -emit-merk-bc %S/Inputs/omp_target_other_module.m -o %t.dir/omp_target_other_module.merkbc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -I %S/Inputs -device-module-dir %t.dir -emit-merk %s -o - | FileCheck %s --check-prefix=COPIED
% COPIED: merk.proc private @omp_target_other_module.triple_f1_0(
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
