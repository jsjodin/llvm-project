% MANUAL: Using foreign enumerations in Mercury code
% A foreign_enum must be for a type the module defines.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_imported_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool.
:- pragma foreign_enum("C", bool/0, [no - "0", yes - "1"]). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
