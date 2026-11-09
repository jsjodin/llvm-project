% A target region sends one-word messages, integers, floats, characters and
% constants of a type of constants only, to mailboxes that main made before
% it, and main receives them after it (docs/INPROGRESS.md, The device's
% side of the first slice, step 2). In the region, process.send is a call
% of the runtime's merk_device_send with the handle's cell, the message
% type's key and the message as a word. The key is a hash of the type's
% name; the host's compile registers each key of the module's sends with
% the host's type_info, in a procedure that runs at start-up, since the
% host encodes the message by it if the mailbox is another node's (step
% 4). On a GPU, merk_device_send is the device's runtime's, which sends it
% over the offload plugin's RPC server to the host, and returns once the
% host has queued it; on the host, it is merkrt's.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_send.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, maybe, process.

:- type colour
    --->    red
    ;       green
    ;       blue.

main(!IO) :-
    new_mailbox(Ints : mailbox(int), !IO),
    new_mailbox(Floats : mailbox(float), !IO),
    new_mailbox(Chars : mailbox(char), !IO),
    new_mailbox(Colours : mailbox(colour), !IO),
    N = 6,
    %$omp target
    (
        send(Ints, N * 7, !IO),
        send(Ints, -1, !IO),
        send(Floats, float(N) / 4.0, !IO),
        send(Chars, 'z', !IO),
        send(Colours, blue, !IO),
        send(Ints, N, !IO)
    ),
    receive_all(Ints, !IO),
    receive_all(Floats, !IO),
    receive_all(Chars, !IO),
    receive_all(Colours, !IO).
% Nothing of the sends crosses but the mailboxes; each send has its type's
% key.
% MERK-LABEL: merk.proc @omp_target_send.main_2_0()
% MERK-NOT: !merk.type_info to !llvm.ptr
% MERK: omp.target
% MERK: %[[W:.*]] = merk.to_word %{{.*}} : i64
% MERK-NEXT: %[[INT:.*]] = arith.constant -7556727498917921977 : i64
% MERK-NEXT: merk.foreign_call @merk_device_send(%{{.*}}, %[[INT]], %[[W]]) : (!merk.term<"process.mailbox(int)">, i64, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send(%{{.*}}, %{{.*}}, %{{.*}}) : (!merk.term<"process.mailbox(float)">, i64, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send(%{{.*}}, %{{.*}}, %{{.*}}) : (!merk.term<"process.mailbox(character)">, i64, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send(%{{.*}}, %{{.*}}, %{{.*}}) : (!merk.term<"process.mailbox(omp_target_send.colour)">, i64, !merk.word) -> ()
% MERK: omp.terminator

% At start-up, each key with its type's type_info, in the keys' order.
% MERK-LABEL: merk.proc private @omp_target_send.merk_register_message_types()
% MERK: merk.type_info "character/0"
% MERK: merk.foreign_call @merk_register_message_type(
% MERK: merk.type_info "float/0"
% MERK: merk.foreign_call @merk_register_message_type(
% MERK: merk.type_info "int/0"
% MERK-NEXT: arith.constant -7556727498917921977 : i64
% MERK: merk.foreign_call @merk_register_message_type(
% MERK: merk.type_info "omp_target_send.colour/0"
% MERK: merk.foreign_call @merk_register_message_type(
% MERK-NEXT: merk.succeed
% MERK: merk.constructor @omp_target_send.merk_register_message_types

% The device calls the device's runtime.
% DEVICE: declare void @merk_device_send(ptr, i64, i64)
% DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_omp_target_send.main_2_0_l{{[0-9]+}}(
% DEVICE-COUNT-6: call void @merk_device_send(

:- pred receive_all(mailbox(T)::in, io::di, io::uo) is det.
receive_all(Mailbox, !IO) :-
    receive(Mailbox, any, after(0), Got, !IO),
    (
        Got = message(M),
        io.print_line(M, !IO),
        receive_all(Mailbox, !IO)
    ;
        Got = timeout
    ).

% OUTPUT:42
% OUTPUT-NEXT:-1
% OUTPUT-NEXT:6
% OUTPUT-NEXT:1.5
% OUTPUT-NEXT:z
% OUTPUT-NEXT:blue
