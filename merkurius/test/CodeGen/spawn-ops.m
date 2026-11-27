% Every spawn is a merk.spawn (docs/INPROGRESS.md, Concurrency in merk,
% step 2 (c1)), which names the body that the process runs, with its
% captured values, and which lowering makes the call it was:
%
% - A named predicate's spawn, with no place, with a place, and with a
%   stack's size: the body is its $words variant, nothing captured.
% - A lambda's spawn, whose captured list was made at run time and
%   survives collections at every allocation.
% - A closure passed in as an argument: a unit body, the closure.
% - A declared kind's spawn at a place: the body is the kind's predicate,
%   with the kind's arguments; the kind's 'spawn_at$Kind' is still called.
%
% mmc has no process module, so the output is written by hand.
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -fprocess-declarations -O0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
%
% CHECK:named: 1
% CHECK-NEXT:at host: 2
% CHECK-NEXT:with a stack: 3
% CHECK-NEXT:lambda: [4, 5, 6]
% CHECK-NEXT:closure: 7
% CHECK-NEXT:kind: 108
% CHECK-NOT:{{.}}
%
% MERK-LABEL: merk.proc @spawn_ops.main_2_0()
% MERK: merk.spawn body @spawn_ops.echo_3_0$words() with %{{.*}} captures [] {merk.closure_entry = @spawn_ops.echo_3_0$words$closure0} : !merk.term<"process.mailbox(T0)">
% MERK: merk.spawn body @spawn_ops.echo_3_0$words() at %{{.*}} : !merk.term<"process.place"> with %{{.*}} captures []
% MERK: merk.spawn body @spawn_ops.echo_3_0$words() stack %{{.*}} with %{{.*}} captures []
% MERK: merk.spawn body @spawn_ops.main_2_0$lambda{{[0-9]+}}(%{{.*}}, %{{.*}} : !merk.term<"process.mailbox(list.list(int))">, !merk.term<"list.list(int)">) with
% MERK: merk.spawn body @spawn_ops.adder_5_0$words(%{{.*}}, %{{.*}} : i64, !merk.term<"process.mailbox(int)">) at %{{.*}} : !merk.term<"process.place"> kind "adder" {merk.spawn_at = @spawn_ops.spawn_at$24adder_6_0} : !merk.term<"process.mailbox(int)">
% MERK-LABEL: merk.proc private @spawn_ops.start_4_0(
% MERK: merk.spawn body unit(%{{.*}} : !merk.closure) with
%
% LLVM-LABEL: define {{.*}} @spawn_ops.main_2_0(
% LLVM: call {{.*}} @process.spawn_4_0(
% LLVM: call {{.*}} @process.spawn_5_0(
% LLVM: call {{.*}} @process.spawn_with_stack_5_0(
% LLVM: call {{.*}} @"spawn_ops.spawn_at$24adder_6_0"(

:- module spawn_ops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, process, string.

:- type request
    --->    request(int, mailbox(int)).

:- process adder(int, mailbox(int)) receives int.

adder(N, Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(M) then
        send(Reply, N + M, !IO)
    else
        true
    ).

    % Sends back the one number it is sent, to the mailbox in the request.
:- pred echo(mailbox(request)::in, io::di, io::uo) is det.
echo(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(request(N, Reply)) then
        send(Reply, N, !IO)
    else
        true
    ).

:- pred start(pred(mailbox(request), io, io)::in(pred(in, di, uo) is det),
    mailbox(request)::out, io::di, io::uo) is det.
start(Body, Box, !IO) :-
    spawn(Body, Box, !IO).

:- pred ask(string::in, mailbox(int)::in, mailbox(request)::in, int::in,
    io::di, io::uo) is det.
ask(Label, Replies, Box, N, !IO) :-
    send(Box, request(N, Replies), !IO),
    receive(Replies, any, after(5000), R, !IO),
    ( if R = message(M) then
        io.format("%s: %d\n", [s(Label), i(M)], !IO)
    else
        io.format("%s: timeout\n", [s(Label)], !IO)
    ).

main(!IO) :-
    new_mailbox(Replies, !IO),
    spawn(echo, Named, !IO),
    ask("named", Replies, Named, 1, !IO),
    Echo = echo,
    spawn(host, Echo, AtHost, !IO),
    ask("at host", Replies, AtHost, 2, !IO),
    spawn_with_stack(1 << 20, echo, Stacked, !IO),
    ask("with a stack", Replies, Stacked, 3, !IO),

    new_mailbox(Lists, !IO),
    Items = list.map(func(X) = X + 3, [1, 2, 3]),
    spawn((pred(_Self::in, IO0::di, IO::uo) is det :-
            send(Lists, Items, IO0, IO)),
        _ : mailbox(int), !IO),
    receive(Lists, any, after(5000), L, !IO),
    ( if L = message(Got) then
        io.write_string("lambda: ", !IO),
        io.write(Got, !IO),
        io.nl(!IO)
    else
        io.write_string("lambda: timeout\n", !IO)
    ),

    start(echo, Started, !IO),
    ask("closure", Replies, Started, 7, !IO),

    spawn(host, adder(100, Replies), Adder, !IO),
    send(Adder, 8, !IO),
    receive(Replies, any, after(5000), K, !IO),
    ( if K = message(Sum) then
        io.format("kind: %d\n", [i(Sum)], !IO)
    else
        io.write_string("kind: timeout\n", !IO)
    ).
