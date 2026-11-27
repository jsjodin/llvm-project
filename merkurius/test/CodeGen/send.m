% send/4 is a merk.send (docs/INPROGRESS.md, Concurrency in merk, step 2),
% which shows the message's own type, and which lowering makes the
% runtime's send on the host, merk_send with the handle's cell, with no
% call of the library's send:
%
% - Messages of one word, a string, and a list made at run time, which
%   survives collections at every allocation.
% - A send in a polymorphic procedure, with the type_info it was given.
% - send passed as a closure stays the library's call, by the closure.
% - A send to a closed mailbox is dropped, as the library's was.
%
% mmc has no process module, so the output is written by hand.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -O0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
%
% CHECK:int: message(7)
% CHECK-NEXT:string: message("seven")
% CHECK-NEXT:list: message([1, 2, 3])
% CHECK-NEXT:polymorphic: message("p")
% CHECK-NEXT:closure: message(1)
% CHECK-NEXT:closure: message(2)
% CHECK-NEXT:closed: done
% CHECK-NOT:{{.}}
%
% MERK-LABEL: merk.proc @send.main_2_0()
% MERK: merk.send %{{.*}} to %{{.*}} with %{{.*}} : i64, !merk.term<"process.mailbox(T0)">
% MERK: merk.send %{{.*}} to %{{.*}} with %{{.*}} : !merk.string, !merk.term<"process.mailbox(T0)">
% MERK-LABEL: merk.proc private @send.forward_4_0(
% MERK: merk.send %{{.*}} to %{{.*}} with %{{.*}} : !merk.word, !merk.term<"process.mailbox(T0)">
%
% LLVM-LABEL: define {{.*}} @send.main_2_0(
% LLVM-NOT: @process.send_4_0(
% LLVM: call void @merk_send(ptr %{{.*}}, ptr @"merk.type_ctor.int/0", i64 7)
% LLVM-NOT: @process.send_4_0(
% LLVM: {{^}$}}

:- module send.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module int, list, process, string.

:- pred forward(mailbox(T)::in, T::in, io::di, io::uo) is det.
forward(Box, Message, !IO) :-
    send(Box, Message, !IO).

:- pred show(string::in, T::in, io::di, io::uo) is det.
show(Label, Value, !IO) :-
    io.write_string(Label ++ ": ", !IO),
    io.write(Value, !IO),
    io.nl(!IO).

main(!IO) :-
    new_mailbox(Ints, !IO),
    send(Ints, 7, !IO),
    receive(Ints, any, after(0), Int, !IO),
    show("int", Int, !IO),

    new_mailbox(Strings, !IO),
    send(Strings, "seven", !IO),
    receive(Strings, any, after(0), String, !IO),
    show("string", String, !IO),

    new_mailbox(Lists, !IO),
    send(Lists, list.map(func(X) = X + 0, [1, 2, 3]), !IO),
    receive(Lists, any, after(0), List, !IO),
    show("list", List, !IO),

    forward(Strings, "p", !IO),
    receive(Strings, any, after(0), Forwarded, !IO),
    show("polymorphic", Forwarded, !IO),

    list.foldl(send(Ints), [1, 2], !IO),
    receive(Ints, any, after(0), First, !IO),
    show("closure", First, !IO),
    receive(Ints, any, after(0), Second, !IO),
    show("closure", Second, !IO),

    close_mailbox(Lists, !IO),
    send(Lists, [4], !IO),
    io.write_string("closed: done\n", !IO).
