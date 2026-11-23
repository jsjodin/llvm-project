% MANUAL: Discriminated unions
% An ordinary discriminated union must be transparent: a type variable of
% the body must occur in the type's name.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_not_transparent.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type box
    --->    box(T).                                 % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
