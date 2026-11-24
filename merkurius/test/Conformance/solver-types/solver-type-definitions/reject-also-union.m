% MANUAL: Solver type definitions
% A solver type may not also be defined by a discriminated union.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_also_union.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t where representation is int.
:- type t ---> a. % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
