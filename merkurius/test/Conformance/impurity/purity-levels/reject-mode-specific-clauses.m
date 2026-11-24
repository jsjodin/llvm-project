% MANUAL: Purity levels
% Clauses for particular modes make a predicate impure unless it is
% promised pure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_mode_specific_clauses.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred f(int, int). % ERROR
:- mode f(in, out) is det.
:- mode f(out, in) is det.
f(X::in, Y::out) :- Y = X + 1.
f(X::out, Y::in) :- X = Y - 1.

main(!IO) :- f(1, A), f(B, 10), io.write([A, B], !IO), io.nl(!IO).
