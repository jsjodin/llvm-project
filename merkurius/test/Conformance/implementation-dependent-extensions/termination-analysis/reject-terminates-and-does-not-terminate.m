% MANUAL: Termination analysis
% A predicate is not both terminates and does_not_terminate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_terminates_and_does_not_terminate.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma terminates(p/2).
:- pragma does_not_terminate(p/2). % ERROR
p(N, N).
main(!IO) :- io.write_string("ok\n", !IO).
