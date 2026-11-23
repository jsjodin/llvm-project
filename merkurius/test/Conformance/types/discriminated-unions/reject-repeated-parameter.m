% MANUAL: Discriminated unions
% The arguments of a polymorphic type's name must be distinct type
% variables.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_repeated_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type twice(T, T)                                 % ERROR
    --->    twice(T).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
