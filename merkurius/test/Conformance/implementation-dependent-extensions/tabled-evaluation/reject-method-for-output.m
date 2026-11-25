% MANUAL: Tabled evaluation
% An output argument's method is output.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_method_for_output.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred q(int, int).
:- mode q(in, out) is det.
:- mode q(out, in) is det.
:- pragma memo(q/2, [specified([value, output])]). % ERROR
q(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
