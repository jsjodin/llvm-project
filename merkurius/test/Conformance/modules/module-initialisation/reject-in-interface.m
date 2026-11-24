% MANUAL: Module initialisation
% initialise directives belong in the implementation section.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:

:- module reject_in_interface.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- initialise hello/2.                              % ERROR
:- implementation.
:- pred hello(io::di, io::uo) is det.
hello(!IO) :-
    io.write_string("hello\n", !IO).
main(!IO) :-
    io.write_string("main\n", !IO).
