% receive/5 on the host is a plain merk.receive, which lowering makes the
% same loop over the runtime's candidates as a receive goal's (Lowering.cpp,
% ExpandReceiveAsSelect, LowerSelect), with no call of the library's
% receive:
%
% - A named selector is called by the loop, giving a value of another type
%   than the message's.
% - A selector that captures a list, made at run time, survives collections
%   at every allocation, as does a list that is live across the receive.
% - A selector passed as an argument is a closure, which the loop calls.
% - any is not called: the loop takes the oldest message.
% - A receive with a type variable's messages, as open.receive's, has the
%   message as its word, which the collector knows by the type_info.
% - A message skipped by many receives is still taken in order.
% - A closed mailbox is the error at once, as the library's receive gave.
%
% mmc has no process module, so the output is written by hand.
%
% RUN: merkc -O0 %s -o %t0 && not %t0 > %t0.out 2> %t0.err
% RUN: FileCheck %s --match-full-lines --strict-whitespace < %t0.out
% RUN: FileCheck %s --check-prefix=ERROR < %t0.err
% RUN: merkc -O2 %s -o %t2 && not %t2 > %t2.out 2> %t2.err
% RUN: FileCheck %s --match-full-lines --strict-whitespace < %t2.out
% RUN: FileCheck %s --check-prefix=ERROR < %t2.err
% RUN: not env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 > %t0.gc.out 2> %t0.gc.err
% RUN: FileCheck %s --match-full-lines --strict-whitespace < %t0.gc.out
% RUN: FileCheck %s --check-prefix=ERROR < %t0.gc.err
% RUN: not env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 > %t2.gc.out 2> %t2.gc.err
% RUN: FileCheck %s --match-full-lines --strict-whitespace < %t2.gc.out
% RUN: FileCheck %s --check-prefix=ERROR < %t2.gc.err
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -O0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM --implicit-check-not='call {{.*}}@process.{{receive_6_0|any_2_0}}'
%
% CHECK:named: message("30")
% CHECK-NEXT:captured: message(9)
% CHECK-NEXT:argument: message(-7)
% CHECK-NEXT:any: message(1)
% CHECK-NEXT:empty: timeout
% CHECK-NEXT:polymorphic: message("a")
% CHECK-NEXT:polymorphic, long: message("ccc")
% CHECK-NEXT:polymorphic: message("bb")
% CHECK-NEXT:skipped: message(150)
% CHECK-NEXT:then: message(1)
% CHECK-NEXT:live: [7, 8, 9]
% CHECK-NOT:{{.}}
%
% ERROR: receive: the mailbox is closed or another node's
%
% MERK-LABEL: merk.proc @receive_plain.main_2_0()
% MERK: merk.receive {{.*}} plain matchers [@receive_plain.small_2_0{{.*}}] counts [0]
% MERK: merk.receive {{.*}} plain matchers [@receive_plain.main_2_0$lambda{{[0-9]+}}] counts [1]
% MERK: merk.receive {{.*}} plain matchers [@process.any_2_0] counts [1]
% MERK-LABEL: merk.proc private @receive_plain.take_5_0(
% MERK: merk.receive {{.*}} plain matchers [unit] counts [1] {{.*}} captured(%{{.*}} : !merk.closure)
% MERK-LABEL: merk.proc private @receive_plain.next_4_0(
% MERK: merk.receive {{.*}} plain matchers [@process.any_2_0] counts [1]
%
% LLVM: call {{.*}}@merk_select_begin(

:- module receive_plain.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, process, string.

:- pred small(int::in, string::out) is semidet.
small(N, string.int_to_string(N * 10)) :-
    N < 5.

:- pred take(mailbox(int)::in, pred(int, int)::in(pred(in, out) is semidet),
    received(int)::out, io::di, io::uo) is det.
take(Box, Selector, Received, !IO) :-
    receive(Box, Selector, after(0), Received, !IO).

:- pred next(mailbox(T)::in, received(T)::out, io::di, io::uo) is det.
next(Box, Received, !IO) :-
    receive(Box, any, after(0), Received, !IO).

:- pred show(string::in, T::in, io::di, io::uo) is det.
show(Label, Value, !IO) :-
    io.write_string(Label ++ ": ", !IO),
    io.write(Value, !IO),
    io.nl(!IO).

:- pred send_range(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.
send_range(Box, From, To, !IO) :-
    ( if From > To then
        true
    else
        send(Box, From, !IO),
        send_range(Box, From + 1, To, !IO)
    ).

main(!IO) :-
    new_mailbox(Box, !IO),
    send(Box, 7, !IO),
    send(Box, 3, !IO),
    send(Box, 9, !IO),
    send(Box, 1, !IO),
    receive(Box, small, after(0), Named, !IO),
    show("named", Named, !IO),
    Wanted = list.map(func(X) = X * 3, [3, 4]),
    receive(Box,
        (pred(N::in, M::out) is semidet :- list.member(N, Wanted), M = N),
        after(0), Captured, !IO),
    show("captured", Captured, !IO),
    take(Box, (pred(N::in, M::out) is semidet :- N mod 2 = 1, M = -N),
        Argument, !IO),
    show("argument", Argument, !IO),
    receive(Box, any, after(0), Any, !IO),
    show("any", Any, !IO),
    receive(Box, any, after(10), Empty, !IO),
    show("empty", Empty, !IO),

    new_mailbox(Words, !IO),
    send(Words, "a", !IO),
    send(Words, "bb", !IO),
    send(Words, "ccc", !IO),
    next(Words, First, !IO),
    show("polymorphic", First, !IO),
    Prefix = string.duplicate_char('c', 2),
    receive(Words,
        (pred(W::in, V::out) is semidet :- string.prefix(W, Prefix), V = W),
        after(0), Long, !IO),
    show("polymorphic, long", Long, !IO),
    next(Words, Second, !IO),
    show("polymorphic", Second, !IO),

    Live = list.map(func(X) = X + 6, [1, 2, 3]),
    send_range(Box, 1, 200, !IO),
    receive(Box, (pred(N::in, M::out) is semidet :- N = 150, M = N),
        after(0), Skipped, !IO),
    show("skipped", Skipped, !IO),
    receive(Box, any, after(0), Then, !IO),
    show("then", Then, !IO),
    show("live", Live, !IO),

    close_mailbox(Box, !IO),
    receive(Box, any, after(0), _, !IO).
