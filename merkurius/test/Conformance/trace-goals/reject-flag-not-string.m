% MANUAL: Trace goals
% A flag's name must be a string.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_flag_not_string.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [compile_time(flag(a)), io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
