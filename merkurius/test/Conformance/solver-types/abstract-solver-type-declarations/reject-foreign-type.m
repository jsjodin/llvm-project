% MANUAL: Abstract solver type declarations
% A solver type declaration may not be defined by a foreign type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_foreign_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- solver type t. % ERROR
:- implementation.
:- import_module int, list.

:- pragma foreign_type("C", t, "MR_Integer").

main(!IO) :- io.write_string("ok\n", !IO).
