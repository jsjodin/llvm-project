% MANUAL: Abstract typeclass declarations
% An abstract typeclass declaration must be accompanied by a non-abstract
% one.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_no_definition.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- typeclass c(T).                                  % ERROR
:- implementation.
main(!IO) :-
    io.write_string("hello\n", !IO).
