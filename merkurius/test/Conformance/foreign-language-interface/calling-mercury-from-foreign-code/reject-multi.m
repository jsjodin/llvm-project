% MANUAL: Calling Mercury from foreign code
% A procedure exported to foreign code may not be multi or nondet.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_multi.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::out) is multi.
p(1).
p(2).
:- pragma foreign_export("C", p(out), "p_from_c"). % ERROR

main(!IO) :- io.write_string("main\n", !IO).
