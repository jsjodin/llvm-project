% MANUAL: Solver type definitions
% A type declared abstractly, not as a solver type, may not have a solver
% type definition.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_declared_not_solver.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- type t. % ERROR
:- implementation.
:- import_module int, list.

:- solver type t where representation is int.

main(!IO) :- io.write_string("ok\n", !IO).
