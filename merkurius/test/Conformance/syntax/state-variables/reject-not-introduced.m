% MANUAL: State variables
% A state variable must be introduced before it is used.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_not_introduced.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::out) is det.
p(X) :- X = !.S. % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
