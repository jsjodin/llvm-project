% MANUAL: Type conversions
% At a recursive node of the inst tree the argument's type must be the
% result's or a subtype of it: a list whose recursive inst binds every
% element to lemon still does not convert from list(fruit) to list(citrus).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:31:{{[0-9]+}}: error:

:- module reject_recursive_inst.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- inst lemons for list/1
    --->    []
    ;       [bound(lemon) | lemons].

:- pred to_citrus(list(fruit)::in(lemons), list(citrus)::out) is det.
to_citrus(L, C) :-
    C = coerce(L).                                      % ERROR

main(!IO) :-
    to_citrus([lemon, lemon], C),
    io.write_line(C, !IO).
