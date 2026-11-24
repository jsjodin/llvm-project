% MANUAL: Solver type definitions
% The attributes must appear in the manual's order.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_bad_attribute_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t where equality is eq, representation is int. % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
