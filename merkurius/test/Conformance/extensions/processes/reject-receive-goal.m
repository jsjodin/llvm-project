% DESIGN: Processes and mailboxes / The receive goal
% A receive goal needs io(!IO); it has at most one timeout arm, which
% needs the parameter timeout(T), and that parameter needs the arm; an
% arm is select(Mailbox, Pattern) -> Goal or timeout -> Goal; and a
% parameter is given once.
% ORACLE: none
%
% RUN: not merkc -fc1 -freceive-goal -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_receive_goal.
:- interface.
:- import_module io, process.
:- type msg ---> a(int) ; b ; c.
:- pred no_io(mailbox(msg)::in, io::di, io::uo) is det.
:- pred two_timeouts(mailbox(msg)::in, io::di, io::uo) is det.
:- pred timeout_arm_only(mailbox(msg)::in, io::di, io::uo) is det.
:- pred timeout_param_only(mailbox(msg)::in, io::di, io::uo) is det.
:- pred not_an_arm(mailbox(msg)::in, io::di, io::uo) is det.
:- pred twice(mailbox(msg)::in, io::di, io::uo) is det.
:- implementation.

no_io(Box, !IO) :-
    receive [timeout(infinity)] (
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:13: error: '[timeout(infinity)]': a receive goal needs the parameter io(!IO), the state that the wait and the arms thread
        select(Box, _) -> true
    ).

two_timeouts(Box, !IO) :-
    receive [timeout(after(1)), io(!IO)] (
        select(Box, _) -> true
    ;   timeout -> true
    ;   timeout -> true
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:9: error: 'timeout -> true': a receive goal has at most one timeout arm
    ).

timeout_arm_only(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, _) -> true
    ;   timeout -> true
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:9: error: 'timeout -> true': a timeout arm needs the parameter timeout(T), which says when it runs
    ).

timeout_param_only(Box, !IO) :-
    receive [timeout(after(1)), io(!IO)] (
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:13: error: '[timeout(after(1)), io(!IO)]': the parameter timeout(T) needs a timeout arm, timeout -> Goal, which runs when no message arrives in time
        select(Box, _) -> true
    ).

not_an_arm(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, _) -> true
    ;   true
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:9: error: 'true' is not a receive arm: expected select(Mailbox, Pattern) -> Goal or timeout -> Goal
    ).

twice(Box, !IO) :-
    receive [io(!IO), io(!IO)] (
% CHECK-DAG: reject-receive-goal.m:[[@LINE-1]]:13: error: the receive parameter 'io' is given more than once
        select(Box, _) -> true
    ).
