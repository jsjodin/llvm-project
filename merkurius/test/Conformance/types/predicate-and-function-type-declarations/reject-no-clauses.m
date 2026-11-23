% MANUAL: Predicate and function type declarations
% Each declared predicate or function needs at least one clause.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:10:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_no_clauses.
:- interface.
:- pred p(int::in) is det.                          % ERROR
:- func f(int) = int.                               % ERROR
:- implementation.
