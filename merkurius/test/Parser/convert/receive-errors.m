% Errors in receive goals, read with -freceive-goal (design document,
% "Processes and mailboxes"). An item with an error is left out.
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-parse-tree-no-sema %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module receive_errors.
:- implementation.

not_arm(B, !IO) :-
    receive [io(!IO)] ( select(B, _) -> true ; q -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:48: error: 'q -> true' is not a receive arm: expected select(Mailbox, Pattern) -> Goal or timeout -> Goal

select_arity(B, !IO) :-
    receive [io(!IO)] ( select(B) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:25: error: 'select(B) -> true' is not a receive arm

no_arrow(B, !IO) :-
    receive [io(!IO)] ( select(B, _), true ).
% CHECK: receive-errors.m:[[@LINE-1]]:25: error: 'select(B, _), true' is not a receive arm

no_io(B) :-
    receive [] ( select(B, _) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:13: error: '[]': a receive goal needs the parameter io(!IO), the state that the wait and the arms thread

io_twice(B, !IO) :-
    receive [io(!IO), io(!IO)] ( select(B, _) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:13: error: the receive parameter 'io' is given more than once

timeout_twice(B, !IO) :-
    receive [io(!IO), timeout(infinity), timeout(after(1))] (
        select(B, _) -> true
    ;   timeout -> true
    ).
% CHECK: receive-errors.m:[[@LINE-4]]:13: error: the receive parameter 'timeout' is given more than once

arm_without_parameter(B, !IO) :-
    receive [io(!IO)] ( select(B, _) -> true ; timeout -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:48: error: 'timeout -> true': a timeout arm needs the parameter timeout(T), which says when it runs

parameter_without_arm(B, !IO) :-
    receive [io(!IO), timeout(infinity)] ( select(B, _) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:13: error: '[io(!IO), timeout(infinity)]': the parameter timeout(T) needs a timeout arm, timeout -> Goal, which runs when no message arrives in time

two_timeout_arms(!IO) :-
    receive [io(!IO), timeout(infinity)] ( timeout -> true ; timeout -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:62: error: 'timeout -> true': a receive goal has at most one timeout arm

bad_parameters(B, !IO) :-
    receive [io(IO0, IO), wait(1)] ( select(B, _) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:14: error: expected a receive parameter: timeout(T) or io(!IO), but found 'io(IO0, IO)'
% CHECK: receive-errors.m:[[@LINE-2]]:27: error: expected a receive parameter: timeout(T) or io(!IO), but found 'wait(1)'

not_a_list(B, !IO) :-
    receive io(!IO) ( select(B, _) -> true ).
% CHECK: receive-errors.m:[[@LINE-1]]:13: error: expected a list of receive parameters, but found 'io(!IO)'

    % Under -freceive-goal, receive is an operator, as trace is, so as a
    % name it needs parentheses.
operand(X) :-
    X = receive.
% CHECK: receive-errors.m:[[@LINE-1]]:9: error: the operator 'receive' must be in parentheses to be used as an operand

in_parentheses(X) :-
    X = (receive).
