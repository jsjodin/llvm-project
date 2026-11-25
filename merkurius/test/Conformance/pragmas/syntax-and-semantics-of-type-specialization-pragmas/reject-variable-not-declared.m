% MANUAL: Syntax and semantics of type specialization pragmas
% A substitution's variables are the declaration's.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_variable_not_declared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(int::in, int::out) is det.
:- pragma type_spec(e/2, T = int). % ERROR
e(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
