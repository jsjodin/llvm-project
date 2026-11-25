% MANUAL: Termination analysis
% A predicate is not both check_termination and terminates.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_check_termination_and_terminates.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma check_termination(p/2).
:- pragma terminates(p/2). % ERROR
p(N, N).
main(!IO) :- io.write_string("ok\n", !IO).
