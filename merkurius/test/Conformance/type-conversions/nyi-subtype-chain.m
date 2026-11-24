% MANUAL: Type conversions
% The base type is found by following a chain of supertypes: a value of a
% subtype of a subtype converts to the base type, and to the type between.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_subtype_chain.lemon_only

:- module nyi_subtype_chain.
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

:- type lemon_only =< citrus
    --->    lemon.

main(!IO) :-
    L = lemon : lemon_only,
    C = coerce(L) : citrus,
    F = coerce(L) : fruit,
    io.write_line(C, !IO),
    io.write_line(F, !IO).
