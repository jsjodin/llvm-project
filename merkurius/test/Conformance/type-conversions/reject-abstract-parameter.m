% MANUAL: Type conversions
% A parameter must be the same in both types if the base type's definition
% has it in an abstract type: holds(T)'s argument is an abs(T), whose
% definition this module cannot see.
% MODULES: Inputs/abstract_holder.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/abstract_holder.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:31:{{[0-9]+}}: error:

:- module reject_abstract_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module abstract_holder.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- type holds(T)
    --->    holds(abs(T)).

main(!IO) :-
    X = holds(make(lemon)) : holds(citrus),
    Y = coerce(X) : holds(fruit),                       % ERROR
    io.write_line(Y, !IO).
