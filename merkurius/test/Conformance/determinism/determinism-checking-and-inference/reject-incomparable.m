% MANUAL: Determinism checking and inference
% A declared determinism not comparable to the inferred one is an error:
% det and failure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_incomparable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred nothing(int::out) is det.                   % ERROR
nothing(_) :-
    fail.

main(!IO) :-
    nothing(Y),
    io.write_line(Y, !IO).
