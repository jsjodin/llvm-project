% A variable of a receive goal's pattern that occurs outside the goal is
% tested, so it must be ground before the goal; one that the arm should
% bind for later goals is an error, which says so (design document, "The
% receive goal").
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module modes.
:- interface.
:- import_module io, process.
:- pred after(mailbox(int)::in, int::out, io::di, io::uo) is det.
:- implementation.

after(Box, N, !IO) :-
    receive [io(!IO)] ( select(Box, N) -> true ).
% CHECK: modes.m:[[@LINE-1]]:25: error: in the clause for after(in, out, di, uo): mode error: N has inst free, but a variable of a receive goal's pattern that occurs outside the goal is tested, so it must be ground before the goal
