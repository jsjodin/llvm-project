% MANUAL: The module system
% The interface section may not contain clauses.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_clause_in_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred p(int::out) is det.
p(1).                                               % ERROR
:- implementation.
main(!IO) :-
    p(X),
    io.write_int(X, !IO).
