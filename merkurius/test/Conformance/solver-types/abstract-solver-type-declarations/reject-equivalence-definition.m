% MANUAL: Abstract solver type declarations
% A solver type declaration may not be defined by an equivalence.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_equivalence_definition.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- solver type t. % ERROR
:- implementation.
:- import_module int, list.

:- type t == int.

main(!IO) :- io.write_string("ok\n", !IO).
