% What the library's code does that a device cannot is an error in the
% device's compile, as in the module's own code, at the library's line, with
% a note for each call in the chain from the region: here list.filter, which
% calls its predicate, a closure (design document, Library code on the
% device). The host's compile is the ordinary one.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=note:
% RUN: merkc -fc1 -fopenmp -emit-merk %s -o /dev/null
% REQUIRES: amdgpu-registered-target

:- module omp_target_library_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    %$omp target
    evens(N),
    io.write_int(N, !IO),
    io.nl(!IO).

:- pred evens(int::out) is det.

evens(N) :-
    N = list.length(list.filter(even, [1, 2, 3, 4])).

% CHECK: omp-target-library-errors.m:[[#@LINE-2]]:33: error: 'merk.closure' op makes a closure, which cannot be done on an OpenMP device yet
% CHECK: list.m:{{[0-9]+}}:{{[0-9]+}}: error: 'merk.call_closure' op calls a closure, which cannot be done on an OpenMP device yet
% CHECK-NEXT: list.m:{{[0-9]+}}:{{[0-9]+}}: note: called by list.filter_f2_0 here: list.filter_3_0
% CHECK-NEXT: omp-target-library-errors.m:[[#@LINE-5]]:21: note: called by omp_target_library_errors.evens_1_0 here: list.filter_f2_0
