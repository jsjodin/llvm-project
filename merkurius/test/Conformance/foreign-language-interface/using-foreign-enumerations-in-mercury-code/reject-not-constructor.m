% MANUAL: Using foreign enumerations in Mercury code
% Each constructor a foreign_enum names must be one of its type's.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_not_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- pragma foreign_enum("C", t/0, [a - "1", b - "2", c - "3"]). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
