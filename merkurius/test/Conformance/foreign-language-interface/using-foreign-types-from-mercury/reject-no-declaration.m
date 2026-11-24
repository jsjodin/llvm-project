% MANUAL: Using foreign types from Mercury
% A foreign_type needs a `:- type' declaration or definition of its type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_no_declaration.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pragma foreign_type("C", t, "long"). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
