% MANUAL: Type conversions
% The manual's examples that are type and mode correct: f1 converts a
% citrus to a fruit, and f3 a fruit whose initial inst limits it to the
% citrus constructors to a citrus.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_manual_examples.citrus

:- module nyi_manual_examples.
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

:- func f1(citrus) = fruit.
f1(X) = coerce(X).

:- inst citrus for fruit/0
    --->    lemon
    ;       orange.

:- func f3(fruit) = citrus.
:- mode f3(in(citrus)) = out is det.
f3(X) = coerce(X).

main(!IO) :-
    io.write_line(f1(lemon), !IO),
    L = lemon : fruit,
    io.write_line(f3(L), !IO),
    io.write_line([f1(orange), apple], !IO).
