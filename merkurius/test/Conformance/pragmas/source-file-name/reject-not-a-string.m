% MANUAL: Source file name
% A source_file pragma's argument is a string.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_not_a_string.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pragma source_file(foo). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
