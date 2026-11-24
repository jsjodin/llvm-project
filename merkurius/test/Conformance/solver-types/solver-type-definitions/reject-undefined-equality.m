% MANUAL: Solver type definitions
% A solver type's equality predicate must be defined.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_undefined_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t where representation is int, equality is eq. % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
