% MANUAL: Solver type definitions
% A solver type may be defined only once.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_defined_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t where representation is int.
:- solver type t where representation is int. % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
