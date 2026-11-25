% MANUAL: Syntax and semantics of type specialization pragmas
% A substitution's types do not mention the variables it replaces.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_both_sides.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(T::in, T::out) is det.
:- pragma type_spec(e/2, T = list(T)). % ERROR
e(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
