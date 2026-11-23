% MANUAL: Different clauses for different modes
% A clause annotated with modes must be for one of the predicate's modes.
% The others make up the procedure, which then has more than one solution.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_clause_for_other_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred same(int, int).
:- mode same(in, out) is det.                       % ERROR
:- pragma promise_equivalent_clauses(same/2).
same(X::in, Y::out) :- Y = X.
same(X::out, Y::in) :- X = Y.                       % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
