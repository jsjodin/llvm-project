% MANUAL: Trace goals
% A grade test must be one of the manual's.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_unknown_grade.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [compile_time(grade(fast)), io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
