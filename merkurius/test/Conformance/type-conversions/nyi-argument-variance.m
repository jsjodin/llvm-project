% MANUAL: Type conversions
% Each type argument of the common base type may be the same in both types,
% a subtype of the other's, or a supertype of it: pair(citrus, fruit) to
% pair(fruit, citrus), whose second argument's inst limits it to citrus
% constructors, and list(citrus) to list(fruit). A coerce between equal
% types is allowed too.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type list.list(nyi_argument_variance.citrus)

:- module nyi_argument_variance.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, pair.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- func up(list(citrus)) = list(fruit).
up(L) = coerce(L).

main(!IO) :-
    P = (lemon - orange) : pair(citrus, fruit),
    Q = coerce(P) : pair(fruit, citrus),
    io.write_line(Q, !IO),
    io.write_line(up([lemon, orange, lemon]), !IO),
    F = apple : fruit,
    G = coerce(F) : fruit,
    io.write_line(G, !IO).
