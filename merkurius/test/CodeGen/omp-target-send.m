% A target region sends one-word messages, integers, floats, characters and
% constants of a type of constants only, to mailboxes that main made before
% it, and main receives them after it (docs/INPROGRESS.md, The device's
% side of the first slice, step 2). In the region, process.send is a call
% of the runtime's merk_device_send with the handle's cell, the message's
% type_info and the message as a word. The type_info is made before the
% region, one for each type, so that the host's reaches the device as one
% of the region's values, mapped by copy: the host encodes the message by
% it if the mailbox is another node's. On a GPU, merk_device_send is the
% device's runtime's, which sends it over the offload plugin's RPC server
% to the host, and returns once the host has queued it; on the host, it is
% merkrt's.
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
% One type_info for each type, made before the region.
% MERK-LABEL: merk.proc @omp_target_send.main_2_0()
% MERK-COUNT-4: merk.call @process.new_mailbox_3_0
% MERK: %[[INT:.*]] = merk.type_info "int/0"
% MERK-NEXT: %[[INTP:.*]] = builtin.unrealized_conversion_cast %[[INT]] : !merk.type_info to !llvm.ptr
% MERK-NEXT: merk.type_info "float/0"
% MERK-NEXT: builtin.unrealized_conversion_cast
% MERK-NEXT: merk.type_info "character/0"
% MERK-NEXT: builtin.unrealized_conversion_cast
% MERK-NEXT: merk.type_info "omp_target_send.colour/0"
% MERK-NEXT: builtin.unrealized_conversion_cast %{{.*}} : !merk.type_info to !llvm.ptr
% MERK-NOT: merk.type_info
% MERK: omp.map.info var_ptr(%{{.*}} : !llvm.ptr, !llvm.ptr) map_clauses(to) capture(ByCopy) name("int/0")
% MERK: omp.target
% MERK: %[[W:.*]] = merk.to_word %{{.*}} : i64
% MERK-NEXT: merk.foreign_call @merk_device_send(%{{.*}}, %{{.*}}, %[[W]]) : (!merk.term<"process.mailbox(int)">, !llvm.ptr, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send({{.*}}) : (!merk.term<"process.mailbox(float)">, !llvm.ptr, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send({{.*}}) : (!merk.term<"process.mailbox(character)">, !llvm.ptr, !merk.word) -> ()
% MERK: merk.foreign_call @merk_device_send({{.*}}) : (!merk.term<"process.mailbox(omp_target_send.colour)">, !llvm.ptr, !merk.word) -> ()
% MERK: omp.terminator

% The device calls the device's runtime.
% DEVICE: declare void @merk_device_send(ptr, ptr, i64)
% DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_omp_target_send.main_2_0_l{{[0-9]+}}(
% DEVICE-COUNT-6: call void @merk_device_send(

:- pred receive_all(mailbox(T)::in, io::di, io::uo) is det.
receive_all(Mailbox, !IO) :-
    receive(Mailbox, any, after(0), Got, !IO),
    (
        Got = yes(M),
        io.print_line(M, !IO),
        receive_all(Mailbox, !IO)
    ;
        Got = no
    ).

% OUTPUT:42
% OUTPUT-NEXT:-1
% OUTPUT-NEXT:6
% OUTPUT-NEXT:1.5
% OUTPUT-NEXT:z
% OUTPUT-NEXT:blue
