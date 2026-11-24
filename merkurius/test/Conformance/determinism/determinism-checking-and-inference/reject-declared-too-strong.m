% MANUAL: Determinism checking and inference
% A declared determinism greater than the inferred one is an error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_declared_too_strong.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred positive(int::in, int::out) is det.         % ERROR
positive(X, X) :-
    X > 0.

main(!IO) :-
    positive(1, Y),
    io.write_line(Y, !IO).
