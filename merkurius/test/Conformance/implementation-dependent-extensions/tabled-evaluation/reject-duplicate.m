% MANUAL: Tabled evaluation
% At most one tabling pragma names a predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_duplicate.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma memo(p/2).
p(X, X).
:- pragma memo(p/2). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
