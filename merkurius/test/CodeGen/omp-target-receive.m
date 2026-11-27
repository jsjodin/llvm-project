% A target region receives: main sends 1 to 10 before it, and the region
% receives and sums them, in a procedure it calls, and sends the sum back;
% then its receive with a timeout of 20 milliseconds finds nothing
% (docs/INPROGRESS.md, The device's side of the first slice, step 3). In
% the region, and in every procedure of the device's compile, which all run
% on the device, receive(Mailbox, any, Timeout, Result, !IO) is a plain
% merk.receive with its ring kind, which lowering makes a call of the
% runtime's merk_device_receive with the handle's cell and the timeout's
% milliseconds, which succeeds with the oldest message as a word.
% On a GPU, merk_device_receive is the device's runtime's, which reads the
% mailbox's ring in host-pinned memory, sleeping between looks, until
% the timeout passes by the device's clock; on the host, it is merkrt's,
% which waits as receive does. A procedure in the host's compile receives
% as on the host. received(int) is a flat union, so message(N) is a value,
% with no cell from merk_device_alloc.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fopenmp --offload-targets=amdgpu-amd-amdhsa -emit-llvm-bc -triple x86_64-unknown-linux-gnu %s -o %t.host.bc
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -fopenmp-host-ir-file-path %t.host.bc -triple amdgpu10.30-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
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
            Late = message(_),
            send(Out, -1, !IO)
        ;
            Late = timeout,
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
% MERK: merk.receive %{{.*}} with %{{.*}} plain matchers [@process.any_2_0] counts [1] on(%{{.*}} : !merk.term<"process.mailbox(int)">) infos(%{{.*}}) captured(%{{.*}} : !merk.type_info) {merk.arms_type = i64, merk.ring_kinds = array<i64: 0>} : !merk.term<"process.timeout"> -> !merk.term<"process.received(int)">
% MERK: omp.terminator

% The device's sum polls too, and makes message(N) as a value.
% DEVICE-LABEL: define internal i64 @omp_target_receive.sum_5_0(
% DEVICE-NOT: merk_device_alloc
% DEVICE: call i32 @merk_device_receive(ptr %{{.*}}, i64 %{{.*}}, ptr %{{.*}})
% DEVICE-NOT: merk_device_alloc
% DEVICE-NOT: process.receive
% DEVICE: {{^}$}}

:- pred sum(mailbox(int)::in, int::in, int::out, io::di, io::uo) is det.
sum(In, !Sum, !IO) :-
    receive(In, any, after(0), Got, !IO),
    (
        Got = message(N),
        !:Sum = !.Sum + N,
        sum(In, !Sum, !IO)
    ;
        Got = timeout
    ).
% The host's sum receives as on the host, with no ring kind.
% MERK-LABEL: merk.proc private @omp_target_receive.sum_5_0(
% MERK: merk.receive %{{.*}} with %{{.*}} plain matchers [@process.any_2_0] counts [1] {{.*}} {merk.arms_type = i64} :

:- pred send_from(int::in, int::in, mailbox(int)::in, io::di, io::uo) is det.
send_from(I, N, M, !IO) :-
    ( if I > N then
        true
    else
        send(M, I, !IO),
        send_from(I + 1, N, M, !IO)
    ).

% OUTPUT:{message(55), message(-2)}
