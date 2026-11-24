% MANUAL: Using foreign types from Mercury
% An equivalence type may not have a foreign_type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_equivalence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t == int.
:- pragma foreign_type("C", t, "long"). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
