% MANUAL: Using Mercury enumerations in foreign code
% Each constructor a foreign_export_enum's overrides name must be one of
% its type's.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_not_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- pragma foreign_export_enum("C", t/0, [], [c - "C"]). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
