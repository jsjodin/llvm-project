% MANUAL: Trace goals
% The io state given back must be the last.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_io_used_after_given.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
main(!IO) :- trace [io(!S)] ( io.write_int(1, !S), io.write_int(2, !.S, _) ), io.nl(!IO). % ERROR
