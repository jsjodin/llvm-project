% MANUAL: Using foreign enumerations in Mercury code
% A type may have at most one foreign_enum for each language.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_two_for_one_language.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- pragma foreign_enum("C", t/0, [a - "1", b - "2"]).
:- pragma foreign_enum("C", t/0, [a - "3", b - "4"]). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
