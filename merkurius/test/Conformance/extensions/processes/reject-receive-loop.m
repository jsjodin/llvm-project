% DESIGN: Processes and mailboxes
% A receive loop declared det must handle every message constructor; one
% that leaves stop out is a determinism error that names it. A receive
% goal's pattern variable that occurs outside the goal is tested, so it
% must be ground before the goal. A receive goal's arms need not cover the
% type.
% ORACLE: none
%
% RUN: not merkc -fc1 -freceive-goal -Wno-unselected-message -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_receive_loop.
:- interface.
:- import_module io, process.
:- type msg ---> increment(int) ; get(mailbox(int)) ; stop.
:- pred loop(int::in, mailbox(msg)::in, io::di, io::uo) is det.
% CHECK-DAG: reject-receive-loop.m:[[@LINE-1]]:9: error: the determinism declaration of loop(in, in, di, uo) is not satisfied: it is declared det, but inferred semidet
:- pred loop_goal(int::in, mailbox(msg)::in, io::di, io::uo) is det.
:- pred unbound_test(mailbox(msg)::in, msg::out, io::di, io::uo) is det.
:- implementation.
:- import_module int.

    % stop is not handled: a det loop must cover every message.
loop(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(Msg),
        (
            Msg = increment(K),
% CHECK-DAG: reject-receive-loop.m:[[@LINE-1]]:13: because: in the case message/1 of the switch on R: the switch on Msg does not cover stop/0
            loop(N + K, Self, !IO)
        ;
            Msg = get(ReplyTo),
            send(ReplyTo, N, !IO),
            loop(N, Self, !IO)
        )
    ;
        R = timeout
    ).

    % A pattern's variable that occurs after the goal is tested, so it
    % must be bound before it.
unbound_test(Self, M, !IO) :-
    receive [io(!IO)] (
        select(Self, M) -> true
% CHECK-DAG: reject-receive-loop.m:[[@LINE-1]]:9: error: in the clause for unbound_test(in, out, di, uo): mode error: M has inst free, but a variable of a receive goal's pattern that occurs outside the goal is tested, so it must be ground before the goal
    ).

    % Not an error: a receive goal's arms need not cover the type.
loop_goal(N, Self, !IO) :-
    receive [io(!IO)] (
        select(Self, increment(K)) -> loop_goal(N + K, Self, !IO)
    ).
