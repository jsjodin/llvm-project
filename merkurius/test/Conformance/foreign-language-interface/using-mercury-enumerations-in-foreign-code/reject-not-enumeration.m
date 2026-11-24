% MANUAL: Using Mercury enumerations in foreign code
% A foreign_export_enum must be for an enumeration type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_not_enumeration.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b(int).
:- pragma foreign_export_enum("C", t/0). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
