% MANUAL: Different clauses for different modes
% A predicate with different clauses for different modes is impure unless
% promise_equivalent_clauses or promise_pure says otherwise, or it is
% declared impure.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_not_promised.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred same(int, int).                             % ERROR
:- mode same(in, out) is det.
:- mode same(out, in) is det.
same(X::in, Y::out) :- Y = X.
same(X::out, Y::in) :- X = Y.

main(!IO) :-
    same(1, Y),
    io.write_line(Y, !IO).
