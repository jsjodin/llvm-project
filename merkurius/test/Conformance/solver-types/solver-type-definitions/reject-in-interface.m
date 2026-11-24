% MANUAL: Solver type definitions
% A solver type may be defined only in the implementation section.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_in_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- solver type t where representation is int. % ERROR
:- implementation.
:- import_module int, list.

main(!IO) :- io.write_string("ok\n", !IO).
