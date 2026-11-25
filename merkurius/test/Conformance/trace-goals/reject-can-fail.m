% MANUAL: Trace goals
% A trace goal must be det or cc_multi: one that cannot succeed.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_can_fail.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det. % ERROR
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [io(!S)] ( 1 = 2, io.write_int(1, !S) ), io.nl(!IO). % ERROR
