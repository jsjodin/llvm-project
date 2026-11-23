% MANUAL: Predicate and function mode declarations
% Each mode declaration of a predicate declares another procedure, so one
% may not be declared twice.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_same_mode_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred p(int, int).
:- mode p(in, out) is det.                          % ERROR
:- mode p(in, out) is det.
p(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
