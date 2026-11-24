% MANUAL: Type conversions
% After replacing each type with its base type, the two types must have the
% same type constructor.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_different_base.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- type veg
    --->    carrot
    ;       leek.

main(!IO) :-
    X = lemon : citrus,
    Y = coerce(X) : veg,                                % ERROR
    io.write_line(Y, !IO).
