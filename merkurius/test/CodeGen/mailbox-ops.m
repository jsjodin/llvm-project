% new_mailbox, close_mailbox, stop, monitor and demonitor are merk
% operations (docs/INPROGRESS.md, Concurrency in merk, step 2 (b)). On the
% host, lowering makes close_mailbox and stop the runtime's merk_close and
% merk_stop, with the handle's cell, and the others the library's calls,
% as a mailbox and a monitor_ref are abstract outside the library:
%
% - A process waiting forever is stopped; its monitor reports it killed.
% - A removed monitor reports nothing.
% - A closed mailbox drops what is sent to it.
%
% process-monitor.m tests monitors at length. mmc has no process module, so
% the output is written by hand.
%
% RUN: merkc -O0 %s -o %t0 && %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t0 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -O0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
%
% CHECK:stopped: message(down(killed))
% CHECK-NEXT:removed: timeout
% CHECK-NEXT:closed: timeout
% CHECK-NOT:{{.}}
%
% MERK-LABEL: merk.proc @mailbox_ops.main_2_0()
% MERK: merk.new_mailbox with %{{.*}} : !merk.term<"process.mailbox(T0)">
% MERK: merk.monitor %{{.*}}, %{{.*}}, %{{.*}} with %{{.*}}, %{{.*}} : !merk.term<"process.mailbox(T0)">, !merk.term<"process.mailbox(T0)"> -> !merk.term<"process.monitor_ref">
% MERK: merk.stop %{{.*}} with %{{.*}} : !merk.term<"process.mailbox(T0)">
% MERK: merk.demonitor %{{.*}} : !merk.term<"process.monitor_ref">
% MERK: merk.close_mailbox %{{.*}} with %{{.*}} : !merk.term<"process.mailbox(T0)">
%
% LLVM-LABEL: define {{.*}} @mailbox_ops.main_2_0(
% LLVM-NOT: @process.{{(stop|close_mailbox)}}_3_0(
% LLVM: call {{.*}} @process.new_mailbox_3_0(
% LLVM: call {{.*}} @process.monitor_6_0(
% LLVM: call void @merk_stop(ptr %{{.*}})
% LLVM: call {{.*}} @process.demonitor_3_0(
% LLVM: call void @merk_close(ptr %{{.*}})
% LLVM-NOT: @process.{{(stop|close_mailbox)}}_3_0(
% LLVM: {{^}$}}

:- module mailbox_ops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module process, string.

:- type happening
    --->    down(down_reason).

:- pred wait(mailbox(int)::in, io::di, io::uo) is det.
wait(Box, !IO) :-
    receive(Box, any, infinity, _, !IO).

:- func went(down_info) = happening.
went(Down) = down(reason(Down)).

:- pred show(string::in, T::in, io::di, io::uo) is det.
show(Label, Value, !IO) :-
    io.write_string(Label ++ ": ", !IO),
    io.write(Value, !IO),
    io.nl(!IO).

main(!IO) :-
    new_mailbox(Events, !IO),
    spawn(wait, Waiting, !IO),
    monitor(Waiting, Events, went, _, !IO),
    stop(Waiting, !IO),
    receive(Events, any, after(5000), Stopped, !IO),
    show("stopped", Stopped, !IO),

    spawn(wait, Other, !IO),
    monitor(Other, Events, went, Ref, !IO),
    demonitor(Ref, !IO),
    stop(Other, !IO),
    receive(Events, any, after(50), Removed, !IO),
    show("removed", Removed, !IO),

    new_mailbox(Box, !IO),
    close_mailbox(Box, !IO),
    send(Box, 1, !IO),
    receive(Events, any, after(0), Closed, !IO),
    show("closed", Closed, !IO).
