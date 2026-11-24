% MANUAL: Type conversions
% A bound inst whose function symbols all name constructors of the result
% type allows a conversion to a subtype; the arguments' insts are checked
% in turn. A value constructed with citrus constructors converts to
% citrus, and a list of them to list(citrus).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_bound_to_subtype.citrus

:- module nyi_bound_to_subtype.
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

:- pred show(citrus::in, io::di, io::uo) is det.
show(C, !IO) :- io.write_line(C, !IO).

main(!IO) :-
    F = orange : fruit,
    show(coerce(F), !IO),
    L = [lemon, orange] : list(fruit),
    C = coerce(L) : list(citrus),
    io.write_line(C, !IO).
