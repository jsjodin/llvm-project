% MANUAL: Subtypes
% A subtype's term is of its supertype only through coerce.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_implicit_upcast.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type fruit
    --->    apple
    ;       lemon
    ;       pear(int).

:- type citrus =< fruit
    --->    lemon.

main(!IO) :-
    D = lemon : citrus,
    eat(D, !IO).                                        % ERROR

:- pred eat(fruit::in, io::di, io::uo) is det.
eat(F, !IO) :-
    io.write_line(F, !IO).
