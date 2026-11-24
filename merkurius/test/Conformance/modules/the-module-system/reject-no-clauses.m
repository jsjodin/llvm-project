% MANUAL: The module system
% The implementation section must define every exported predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_no_clauses.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred p(int::out) is det.                         % ERROR
:- implementation.
main(!IO) :-
    p(X),
    io.write_int(X, !IO).
