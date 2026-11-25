% MANUAL: Trace goals
% A trace goal may not bind a variable of the surrounding code.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_binds_outside.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [io(!S)] ( X = 1, io.write_int(X, !S) ), io.write_int(X, !IO). % ERROR
