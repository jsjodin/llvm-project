% MANUAL: Using foreign types from Mercury
% A subtype may not have a foreign_type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_subtype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- type s =< t ---> a.
:- pragma foreign_type("C", s, "long"). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
