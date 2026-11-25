% DESIGN: Processes and mailboxes / The receive goal
% A mailbox whose arms leave constructors out draws the warning
% unselected-message, once per predicate and mailbox, listing them, unless
% another receive in the predicate selects them. An arm whose pattern is a
% variable of its own selects every message; a tested variable selects no
% constructor.
% ORACLE: none
%
% RUN: merkc -fc1 -freceive-goal -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
% RUN: merkc -fc1 -freceive-goal -Wno-unselected-message -emit-merk %s -o /dev/null 2>&1 | count 0
:- module warn_unselected.
:- interface.
:- import_module io, process.
:- type msg ---> a(int) ; b ; c(string).
:- pred partial(mailbox(msg)::in, io::di, io::uo) is det.
:- pred elsewhere(mailbox(msg)::in, io::di, io::uo) is det.
:- pred every(mailbox(msg)::in, io::di, io::uo) is det.
:- pred tested(mailbox(msg)::in, msg::in, io::di, io::uo) is det.
:- implementation.

    % b and c are left out.
partial(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, a(_)) -> true
% CHECK-DAG: warn-unselected.m:[[@LINE-1]]:16: warning: no arm selects the messages of the mailbox Box whose functor is one of the constructors b/0, c/1, so they stay queued for another receive [-Wunselected-message]
    ).

    % Another receive of the predicate selects b and c.
elsewhere(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, a(1)) -> true
    ),
    receive [io(!IO)] (
        select(Box, b) -> true
    ;   select(Box, c(_)) -> true
    ).

    % A variable of the arm's own selects every message.
every(Box, !IO) :-
    receive [io(!IO)] (
        select(Box, a(_)) -> true
    ;   select(Box, _) -> true
    ).

    % A tested variable selects no constructor: all three are left out.
tested(Box, M, !IO) :-
    receive [io(!IO)] (
        select(Box, M) -> true
% CHECK-DAG: warn-unselected.m:[[@LINE-1]]:16: warning: no arm selects the messages of the mailbox Box whose functor is one of the constructors a/1, b/0, c/1, so they stay queued for another receive [-Wunselected-message]
    ).
