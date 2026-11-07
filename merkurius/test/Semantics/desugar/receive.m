% Receive goals, with -freceive-goal (design document, "Processes and
% mailboxes"), are reported, and their clause is left out.
%
% Temporary (receive goal step 2, docs/INPROGRESS.md): desugaring makes
% each receive goal a receive over a selector of its arms, which this test
% will check instead.
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-semantic-form %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module receive_goals.
:- interface.
:- import_module io.

:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module process.

main(!IO) :-
    new_mailbox(Box, !IO),
    send(Box, 1, !IO),
    receive [io(!IO)] ( select(Box, N) -> io.write_int(N, !IO) ).
% CHECK: receive.m:[[@LINE-1]]:5: error: not yet implemented: desugaring receive goals
