% MANUAL: Adding foreign declarations
% A foreign_decl may not be in the interface section.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pragma foreign_decl("C", "#define X 1"). % ERROR
:- implementation.

main(!IO) :- io.write_string("main\n", !IO).
