% MANUAL: Using foreign types from Mercury
% A type with a foreign_type may not be the supertype of a subtype.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_supertype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- pragma foreign_type("C", t, "long").
:- type s =< t ---> a. % ERROR

main(!IO) :- io.write_string("main\n", !IO).
