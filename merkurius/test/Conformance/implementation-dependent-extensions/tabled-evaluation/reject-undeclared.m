% MANUAL: Tabled evaluation
% A tabling pragma names a predicate or function of the module.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_undeclared.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma memo(nosuch/2). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
