% MANUAL: Conditional expressions
% A conditional expression has an else part.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_missing_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :-
    X = ( if 1 < 2 then 1 ), % ERROR
    io.write_int(X, !IO).
