% A receive that can never get a message is a run-time error, at once, and is
% not reported as a timeout: a closed mailbox, a selector whose mailboxes are all
% closed or that has none, and a receive goal with a timeout arm. A selector
% with an open mailbox among closed ones is not an error. Each error ends a
% process of its own, whose monitor reports it. Another node's mailbox is in
% process-remote.m.
%
% RUN: merkc -freceive-goal -O0 %s -o %t0 && %t0 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -freceive-goal -O2 %s -o %t2 && %t2 2>/dev/null | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

:- module process_receive_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

:- pred closed_mailbox(io::di, io::uo) is det.
closed_mailbox(!IO) :-
    new_mailbox(Box : mailbox(int), !IO),
    close_mailbox(Box, !IO),
    receive(Box, any, after(1000), _, !IO).

:- func same(int) = int.
same(N) = N.

:- pred closed_selector(io::di, io::uo) is det.
closed_selector(!IO) :-
    new_mailbox(A, !IO),
    new_mailbox(B, !IO),
    close_mailbox(A, !IO),
    close_mailbox(B, !IO),
    receive(selecting(A, same, selecting(B, same, new_selector)), after(1000),
        _, !IO).

:- pred no_cases(io::di, io::uo) is det.
no_cases(!IO) :-
    receive(new_selector : selector(int), after(1000), _, !IO).

:- pred goal_with_timeout_arm(io::di, io::uo) is det.
goal_with_timeout_arm(!IO) :-
    new_mailbox(Box : mailbox(int), !IO),
    close_mailbox(Box, !IO),
    receive [timeout(after(1000)), io(!IO)] (
        select(Box, _) ->
            true
    ;
        timeout ->
            true
    ).

:- pred prober(pred(io, io)::in(pred(di, uo) is det), mailbox(int)::in,
    io::di, io::uo) is det.
prober(Goal, Own, !IO) :-
    receive(Own, any, infinity, _, !IO),
    Goal(!IO).

:- pred probe(pred(io, io)::in(pred(di, uo) is det), io::di, io::uo) is det.
probe(Goal, !IO) :-
    new_mailbox(Downs, !IO),
    spawn(prober(Goal), Prober, !IO),
    monitor(Prober, Downs, reason, _, !IO),
    send(Prober, 0, !IO),
    receive(Downs, any, after(10000), Got, !IO),
    (
        Got = message(Reason),
        io.write_string(reason_text(Reason), !IO),
        io.nl(!IO)
    ;
        Got = timeout,
        io.write_string("the prober did not end\n", !IO)
    ).

main(!IO) :-
    probe(closed_mailbox, !IO),
    probe(closed_selector, !IO),
    probe(no_cases, !IO),
    probe(goal_with_timeout_arm, !IO),
    new_mailbox(Closed, !IO),
    new_mailbox(Open, !IO),
    close_mailbox(Closed, !IO),
    send(Open, 1, !IO),
    Selector = selecting(Closed, same, selecting(Open, same, new_selector)),
    receive(Selector, after(0), First, !IO),
    io.write_line(First, !IO),
    receive(Selector, after(0), Second, !IO),
    io.write_line(Second, !IO).

% OUTPUT:Software Error: receive: the mailbox is closed or another node's
% OUTPUT-NEXT:Software Error: receive: every mailbox is closed or another node's
% OUTPUT-NEXT:Software Error: receive: every mailbox is closed or another node's
% OUTPUT-NEXT:Software Error: receive: every mailbox is closed or another node's
% OUTPUT-NEXT:message(1)
% OUTPUT-NEXT:timeout
% OUTPUT-NOT:{{.}}
