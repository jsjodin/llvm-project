% MANUAL: Tabled evaluation
% A specified attribute gives a method for each argument.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_too_few_methods.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::in, int::out) is det.
:- pragma memo(p/3, [specified([value, addr])]). % ERROR
p(X, Y, X + Y).
main(!IO) :- io.write_string("ok\n", !IO).
