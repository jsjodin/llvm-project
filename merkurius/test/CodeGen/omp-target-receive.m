% A target region receives: main sends 1 to 10 before it, and the region
% receives and sums them, in a procedure it calls, and sends the sum back;
% then its receive with a timeout of 20 milliseconds finds nothing
% (docs/INPROGRESS.md, The device's side of the first slice, step 3). In
% the region, and in every procedure of the device's compile, which all run
% on the device, receive(Mailbox, any, Timeout, Result, !IO) is a call of
% the runtime's merk_device_receive with the handle's cell and the
% timeout's milliseconds, which succeeds with the oldest message as a word.
% On a GPU, merk_device_receive is the device's runtime's, which polls the
% host over the offload plugin's RPC server, sleeping between polls, until
% the timeout passes by the device's clock; on the host, it is merkrt's,
% which waits as receive does. A procedure in the host's compile receives
% as on the host. yes(N) is a cell, which on the device comes from
% merk_device_alloc.
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

:- module omp_target_receive.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

main(!IO) :-
    new_mailbox(In : mailbox(int), !IO),
    new_mailbox(Out : mailbox(int), !IO),
    send_from(1, 10, In, !IO),
    %$omp target
    (
        sum(In, 0, Sum, !IO),
        send(Out, Sum, !IO),
        receive(In, any, after(20), Late, !IO),
        (
            Late = yes(_),
            send(Out, -1, !IO)
        ;
            Late = no,
            send(Out, -2, !IO)
        )
    ),
    receive(Out, any, after(0), R1, !IO),
    receive(Out, any, after(0), R2, !IO),
    io.print_line({R1, R2}, !IO).
% The timeout's milliseconds, and the receive; any's closure is not made.
% MERK-LABEL: merk.proc @omp_target_receive.main_2_0()
% MERK: omp.target
% MERK-NOT: merk.closure
% MERK: %[[MS:.*]] = merk.switch %{{.*}} : !merk.term<"process.timeout"> -> i64 cases {{\[\[}}#merk.functor<"infinity", 0>], [#merk.functor<"after", 1>]] {
% MERK-NEXT: %[[INF:.*]] = arith.constant -1 : i64
% MERK-NEXT: merk.yield %[[INF]] : i64
% MERK-NEXT: }, {
% MERK-NEXT: %[[N:.*]] = merk.deconstruct #merk.functor<"after", 1> %{{.*}} : !merk.term<"process.timeout"> -> i64
% MERK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
% MERK-NEXT: %[[MAX:.*]] = arith.maxsi %[[N]], %[[ZERO]] : i64
% MERK-NEXT: merk.yield %[[MAX]] : i64
% MERK-NEXT: }
% MERK-NEXT: %[[GOT:.*]]:2 = merk.foreign_call @merk_device_receive(%{{.*}}, %[[MS]]) can_fail : (!merk.term<"process.mailbox(int)">, i64) -> (i1, !merk.word)
% MERK-NEXT: merk.ite -> !merk.term<"maybe.maybe(int)"> {
% MERK-NEXT: merk.guard %[[GOT]]#0
% MERK-NEXT: merk.yield
% MERK-NEXT: } then {
% MERK-NEXT: %[[M:.*]] = merk.from_word %[[GOT]]#1 : i64
% MERK-NEXT: %[[YES:.*]] = merk.construct #merk.functor<"yes", 1>(%[[M]] : i64) -> !merk.term<"maybe.maybe(int)">
% MERK-NEXT: merk.yield %[[YES]]
% MERK-NEXT: } else {
% MERK-NEXT: %[[NO:.*]] = merk.construct #merk.functor<"no", 0> -> !merk.term<"maybe.maybe(int)">
% MERK-NEXT: merk.yield %[[NO]]
% MERK-NEXT: }
% MERK: omp.terminator

% The device's sum polls too, and makes yes(N) with merk_device_alloc.
% DEVICE-LABEL: define internal i64 @omp_target_receive.sum_5_0(
% DEVICE: call i32 @merk_device_receive(ptr %{{.*}}, i64 %{{.*}}, ptr %{{.*}})
% DEVICE: call ptr @merk_device_alloc(i64 8)
% DEVICE-NOT: process.receive

:- pred sum(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.
sum(In, !Sum, !IO) :-
    receive(In, any, after(0), Got, !IO),
    (
        Got = yes(N),
        !:Sum = !.Sum + N,
        sum(In, !Sum, !IO)
    ;
        Got = no
    ).
% The host's sum receives as on the host.
% MERK-LABEL: merk.proc private @omp_target_receive.sum_5_0(
% MERK: merk.call @process.receive_6_0(

:- pred send_from(int::in, int::in, mailbox(int)::in, io::di, io::uo) is det.
send_from(I, N, M, !IO) :-
    ( if I > N then
        true
    else
        send(M, I, !IO),
        send_from(I + 1, N, M, !IO)
    ).

% OUTPUT:{yes(55), yes(-2)}
