% MANUAL: Abstract solver type declarations
% An abstract solver type needs a definition.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_no_definition.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- solver type t. % ERROR
:- implementation.
:- import_module int, list.

main(!IO) :- io.write_string("ok\n", !IO).
