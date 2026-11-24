% MANUAL: Solver type definitions
% A solver type may not have constructors.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_constructors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- solver type t ---> a ; b. % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
