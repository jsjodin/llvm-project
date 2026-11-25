% MANUAL: Trace goals
% A trace level must be shallow or deep.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_unknown_trace_level.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [compile_time(tracelevel(medium)), io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
