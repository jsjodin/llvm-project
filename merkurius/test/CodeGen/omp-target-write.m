% Output from a target region, directly and in a procedure that it calls: on a
% device, io's procedures, copied from the library with their C, which there
% calls the GPU's libc, whose stdio writes into the host's stdout
% (design document, Library code on the device). On the host, a
% region calls io's too.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
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

% DEVICE-DAG: define internal void @io.write_string_3_0(
% DEVICE-DAG: call i32 @fputs(
% DEVICE-DAG: call i32 (ptr, ...) @printf(
% DEVICE-DAG: call i32 @putc(i32 noundef 10,

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
