% Forms that desugaring does not handle yet are reported, and their clauses
% left out. The module compiles with mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 | FileCheck %s

:- module not_yet.
:- interface.
:- import_module io, list.

:- pred main(io::di, io::uo) is det.
:- func add_one(list(int)) = list(int).

:- implementation.
:- import_module int.

main(!IO) :-
    io.write_string("hello\n", !IO).

add_one(Xs) = Ys :-
    Ys = list.map(func(X) = X + 1, Xs).
% CHECK: not-yet.m:[[@LINE-1]]:19: error: not yet implemented: desugaring lambda expressions
