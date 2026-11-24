% MANUAL: Committed choice nondeterminism
% A call to a committed choice mode must be in a single-solution context;
% the body of a multi procedure is not one.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_outside_single_solution.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred digit(int::out) is cc_multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

:- pred digits(int::out) is multi.
digits(X) :-
    digit(X).                                       % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
