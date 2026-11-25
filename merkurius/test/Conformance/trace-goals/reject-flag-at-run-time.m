% MANUAL: Trace goals
% A run-time condition must be env.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_flag_at_run_time.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [run_time(flag("a")), io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
