% A mailbox handle crosses into a target region as a cell of five words,
% process.mailbox's layout, which code generation knows as it knows an
% array's, since the type is abstract outside the library. The region only
% passes it on so far; sending and receiving on the device come next
% (docs/INPROGRESS.md, The device's side of the first slice).
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-merk %s | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o /dev/null
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_mailbox.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- func weight(mailbox(int), int) = int.
weight(_, N) = N * 2.

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    W = weight(Mailbox, 21),
    send(Mailbox, W, !IO),
    receive(Mailbox, any, after(0), Got, !IO),
    (
        Got = message(N),
        io.write_string("got ", !IO),
        io.write_int(N, !IO),
        io.nl(!IO)
    ;
        Got = timeout,
        io.write_string("nothing\n", !IO)
    ).
% MERK-LABEL: merk.proc @omp_target_mailbox.main_2_0()
% MERK: %[[B:.*]] = omp.map.bounds lower_bound(%[[L:.*]] : i64) upper_bound(%[[U:.*]] : i64) extent(%[[E:.*]] : i64)
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) bounds(%[[B]]) name("Mailbox")
% MERK: omp.target
% MERK: builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr to !merk.term<"process.mailbox(int)">

% DEVICE-LABEL: merk.proc @omp_target_mailbox.main_2_0()
% DEVICE: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, i64) map_clauses(to) capture(ByRef) name("Mailbox")

% OUTPUT:got 42
