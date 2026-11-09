% Output from a target region, directly and in a procedure that it calls: on a
% device, calls of the runtime's merk_device_write_ functions, which have the
% host print. On the host, a region calls io's.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.optimized && %libomp-env %t.optimized | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_write.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

% MERK-DAG: merk.foreign_call @merk_device_write_string(
% MERK-DAG: merk.foreign_call @merk_device_write_int(
% MERK-DAG: merk.foreign_call @merk_device_nl()

:- pred line(string::in, int::in, io::di, io::uo) is det.
line(Text, N, !IO) :-
    io.write_string(Text, !IO),
    io.write_int(N, !IO),
    io.nl(!IO).

main(!IO) :-
    N = 41,
    %$omp target
    (
        io.write_string("in the region\n", !IO),
        io.write_string("", !IO),
        line("a long line, longer than one buffer of the RPC protocol, which is fifty-six bytes: ", N + 1, !IO),
        line("negative: ", -N, !IO)
    ),
    io.write_string("after the region\n", !IO).

% OUTPUT:in the region
% OUTPUT-NEXT:a long line, longer than one buffer of the RPC protocol, which is fifty-six bytes: 42
% OUTPUT-NEXT:negative: -41
% OUTPUT-NEXT:after the region
