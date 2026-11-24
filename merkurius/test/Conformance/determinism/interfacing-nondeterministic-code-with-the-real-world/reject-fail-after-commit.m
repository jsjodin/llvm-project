% MANUAL: Interfacing nondeterministic code with the real world
% No goal after a committed choice call in a conjunction may fail, as that
% would ask the call for a second solution.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:25:{{[0-9]+}}: error:

:- module reject_fail_after_commit.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred digit(int::out) is multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

:- pred any_digit(int::out) is cc_multi.
any_digit(X) :-
    digit(X).

:- pred small_digit(int::out) is cc_nondet.
small_digit(D) :-
    any_digit(D),                                   % ERROR
    D < 2.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
