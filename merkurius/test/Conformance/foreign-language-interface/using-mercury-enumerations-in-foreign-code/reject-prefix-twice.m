% MANUAL: Using Mercury enumerations in foreign code
% "A pragma foreign_export_enum declaration may contain at most one prefix
% attribute."
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_prefix_twice.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> red ; green ; blue.
:- pragma foreign_export_enum("C", t/0, [prefix("A_"), prefix("B_")]). % ERROR

main(!IO) :- io.write_string("ok\n", !IO).
