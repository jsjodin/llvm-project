% MANUAL: Tabled evaluation
% A tabling pragma's modes are a mode of the predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_undeclared_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma memo(p(out, in)). % ERROR
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
