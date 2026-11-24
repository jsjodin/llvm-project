% MANUAL: Type conversions
% A parameter must be the same in both types if the base type's definition
% has it in another discriminated union type: box(T)'s argument is a
% wrap(T).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:31:{{[0-9]+}}: error:

:- module reject_non_recursive_parameter.
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

:- type wrap(T)
    --->    wrap(T).

:- type box(T)
    --->    box(wrap(T)).

main(!IO) :-
    X = box(wrap(lemon)) : box(citrus),
    Y = coerce(X) : box(fruit),                         % ERROR
    io.write_line(Y, !IO).
