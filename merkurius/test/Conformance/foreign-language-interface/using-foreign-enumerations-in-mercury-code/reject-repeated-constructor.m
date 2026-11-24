% MANUAL: Using foreign enumerations in Mercury code
% A foreign_enum may not give a constructor two values.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_repeated_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type t ---> a ; b.
:- pragma foreign_enum("C", t/0, [a - "1", a - "2", b - "3"]). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
