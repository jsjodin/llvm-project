% MANUAL: Interfacing nondeterministic code with the real world
% Calling a multi predicate, using its outputs, from a det one is a
% determinism error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_multi_in_det.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred digit(int::out) is multi.
digit(X) :-
    ( X = 3 ; X = 1 ; X = 2 ).

:- pred first(int::out) is det.                     % ERROR
first(D) :-
    digit(D).

main(!IO) :-
    first(D),
    io.write_line(D, !IO).
