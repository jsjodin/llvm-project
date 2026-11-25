% MANUAL: Tabled evaluation
% At most one of memo, loop_check and minimal_model names a predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_two_kinds.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma memo(p/2).
:- pragma loop_check(p/2). % ERROR
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
