% MANUAL: Trace goals
% A trace parameter must be one of the four kinds.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_unknown_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [verbose, io(!S)] ( io.write_int(1, !S) ), io.nl(!IO). % ERROR
