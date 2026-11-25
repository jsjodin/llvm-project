% MANUAL: Goals
% arbitrary goals occur only inside promise_equivalent_solution_sets.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_arbitrary_outside.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred any_of(int::out) is multi.
any_of(1).
any_of(2).
main(!IO) :-
    arbitrary [A] any_of(A), % ERROR
    io.write_int(A, !IO).
