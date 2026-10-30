% Forms that desugaring does not handle yet are reported, and their clauses
% left out. The module compiles with mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 | FileCheck %s

:- module not_yet.
:- interface.
:- import_module io.

:- pred main(io::di, io::uo) is det.
:- pred safe_write(int::in, io::di, io::uo) is cc_multi.

:- implementation.
:- import_module exception.

main(!IO) :-
    io.write_string("hello\n", !IO).

safe_write(X, !IO) :-
    ( try [io(!IO)] io.write_int(X, !IO)
    then true
    catch_any _ -> io.write_string("failed\n", !IO)
    ).
% CHECK: not-yet.m:[[@LINE-4]]:7: error: not yet implemented: desugaring try goals
