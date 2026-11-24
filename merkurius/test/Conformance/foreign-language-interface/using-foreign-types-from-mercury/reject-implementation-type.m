% MANUAL: Using foreign types from Mercury
% A type declared in the implementation section must have its foreign_type
% there too.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_implementation_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pragma foreign_type("C", t, "long"). % ERROR
:- implementation.
:- type t ---> t.

main(!IO) :- io.write_string("main\n", !IO).
