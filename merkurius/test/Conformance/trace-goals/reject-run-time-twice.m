% MANUAL: Trace goals
% The run_time parameter may appear only once.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_run_time_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [run_time(env("A")), run_time(env("B")), io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
