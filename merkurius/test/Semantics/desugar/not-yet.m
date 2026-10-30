% Forms that desugaring does not handle yet are reported, and their clauses
% left out. The module compiles with mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-semantic-form %s 2>&1 | FileCheck %s

:- module not_yet.
:- interface.
:- import_module char, io, list.

:- pred main(io::di, io::uo) is det.
:- pred digits(list(char)::out, list(char)::in, list(char)::out) is multi.

:- implementation.

main(!IO) :-
    io.write_string("hello\n", !IO).

digits([D | Ds]) -->
    [D],
    { char.is_digit(D) },
    digits(Ds).
% CHECK: not-yet.m:[[@LINE-4]]:1: error: not yet implemented: desugaring DCG rules
digits([]) --> [].
% CHECK: not-yet.m:[[@LINE-1]]:1: error: not yet implemented: desugaring DCG rules
