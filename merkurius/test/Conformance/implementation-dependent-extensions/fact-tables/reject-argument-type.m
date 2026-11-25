% MANUAL: Fact tables
% A fact table's arguments are of the types int, float and string.
% FILES: Inputs/ages
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_argument_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred age(list(int)::in, int::out) is semidet. % ERROR
:- pragma fact_table(age/2, "ages").
main(!IO) :- io.write_string("ok\n", !IO).
