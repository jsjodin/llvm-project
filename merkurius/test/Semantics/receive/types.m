% A pattern of a receive goal has the type of its mailbox's messages
% (design document, "The receive goal").
%
% RUN: not merkc -fc1 -freceive-goal -fdebug-dump-determinism %s 2>&1 \
% RUN:   | FileCheck %s --implicit-check-not=error:

:- module types.
:- interface.
:- import_module io, process.
:- pred typed(mailbox(int)::in, io::di, io::uo) is det.
:- implementation.

typed(Box, !IO) :-
    receive [io(!IO)] ( select(Box, "text") -> true ).
% CHECK: types.m:[[@LINE-1]]:37: error: type error in unification of variable Message and `"text"': variable Message has type int, but `"text"' has type string
