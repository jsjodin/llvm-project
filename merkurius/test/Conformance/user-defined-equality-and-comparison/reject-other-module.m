% MANUAL: User-defined equality and comparison
% The equality predicate must be defined in the same module as the type.
% MODULES: Inputs/box_equality.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/box_equality.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_other_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module box_equality.

:- type box                                         % ERROR
    --->    box(int)
            where equality is any_equal.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
