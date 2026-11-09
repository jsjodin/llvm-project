% A send in a procedure that a target region calls, rather than in the
% region itself. In the device's compile every procedure runs on the
% device, so its sends are the device's, with the message type's key, which
% the host's compile registers at start-up (docs/INPROGRESS.md, The
% device's side of the first slice, step 4). In the host's compile the
% procedure sends as on the host.
%
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fc1 -fopenmp -S -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=ASM
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_send_call.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pred twice(mailbox(int)::in, int::in, io::di, io::uo) is det.
twice(Mailbox, N, !IO) :-
    send(Mailbox, N, !IO),
    send(Mailbox, N + 1, !IO).
% DEVICE-LABEL: define internal void @omp_target_send_call.twice_4_0(
% DEVICE-COUNT-2: call void @merk_device_send(ptr %{{.*}}, i64 -7556727498917921977, i64 %{{.*}})
% MERK-LABEL: merk.proc private @omp_target_send_call.twice_4_0(
% MERK-COUNT-2: merk.call @process.send_4_0(
% MERK-NOT: merk_device_send
% MERK-LABEL: merk.proc private @omp_target_send_call.merk_register_message_types()
% MERK: merk.type_info "int/0"
% MERK-NEXT: arith.constant -7556727498917921977 : i64
% MERK: merk.constructor @omp_target_send_call.merk_register_message_types
% The constructor is a C function that calls the procedure, which
% llvm.global_ctors lists, in .init_array, which dynamic loaders run.
% LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @omp_target_send_call.merk_register_message_types.c, ptr null }]
% LLVM: define internal void @omp_target_send_call.merk_register_message_types.c() {
% LLVM-NEXT: call tailcc void @omp_target_send_call.merk_register_message_types()
% LLVM-NEXT: ret void
% ASM: .section .init_array,"aw",@init_array
% ASM-NEXT: .p2align
% ASM-NEXT: .quad omp_target_send_call.merk_register_message_types.c

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    twice(Mailbox, 5, !IO),
    receive(Mailbox, any, after(0), A, !IO),
    receive(Mailbox, any, after(0), B, !IO),
    io.print_line({A, B}, !IO).

% OUTPUT:{message(5), message(6)}
