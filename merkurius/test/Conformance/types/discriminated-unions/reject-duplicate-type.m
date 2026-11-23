% MANUAL: Discriminated unions
% Two definitions of a type with the same name and arity in one module are
% an error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_duplicate_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type colour
    --->    red
    ;       green.

:- type colour                                      % ERROR
    --->    blue.

main(!IO) :-
    io.write_line(red, !IO).
