% MANUAL: pragma foreign_proc
% A predicate of two modes with clauses and a foreign_proc for one is
% impure unless promised pure or promise_equivalent_clauses, as for clauses
% of each mode.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_mixed_impure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.
:- pred inc(int, int). % ERROR
:- mode inc(in, out) is det.
:- mode inc(out, in) is det.
:- pragma foreign_proc("C", inc(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X + 1;").
inc(X, Y) :- X = Y - 1.

main(!IO) :- inc(1, A), inc(B, 10), io.write([A, B], !IO), io.nl(!IO).
