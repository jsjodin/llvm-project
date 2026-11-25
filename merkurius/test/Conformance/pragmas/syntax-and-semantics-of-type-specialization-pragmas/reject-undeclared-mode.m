% MANUAL: Syntax and semantics of type specialization pragmas
% A type_spec pragma's modes are a mode the predicate has.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_undeclared_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(T::in, T::out) is det.
:- pragma type_spec(e(out, in), T = int). % ERROR
e(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
