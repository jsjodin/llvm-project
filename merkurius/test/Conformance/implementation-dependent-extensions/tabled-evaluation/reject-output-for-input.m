% MANUAL: Tabled evaluation
% An input argument's method is value, addr or promise_implied.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_output_for_input.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma memo(p/2, [specified([output, value])]). % ERROR
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
