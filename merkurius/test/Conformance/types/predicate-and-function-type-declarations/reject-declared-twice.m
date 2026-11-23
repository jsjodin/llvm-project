% MANUAL: Predicate and function type declarations
% A predicate or function may be declared only once in a module.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_declared_twice.
:- interface.
:- pred p(int::in) is det.
:- pred p(int::in) is det.                          % ERROR
:- func f(int) = int.
:- func f(int) = int.                               % ERROR
:- implementation.
p(_).
f(X) = X.
