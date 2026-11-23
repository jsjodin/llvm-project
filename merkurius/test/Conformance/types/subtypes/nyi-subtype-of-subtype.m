% MANUAL: Subtypes
% A supertype may itself be a subtype, back to a base type; its arguments
% need not be type variables, nor distinct ones; a subtype may have a
% phantom parameter; and a subtype exported abstractly is defined in the
% implementation section. A switch on a subtype need cover only its own
% constructors.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_subtype_of_subtype.warm

:- module nyi_subtype_of_subtype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- type nel(T).

:- implementation.
:- import_module int, list, pair, string.

:- type nel(T) =< list(T)
    --->    [T | list(T)].

:- type colour
    --->    red
    ;       green
    ;       blue
    ;       black.

:- type bright =< colour
    --->    red
    ;       green
    ;       blue.

:- type warm =< bright
    --->    red.

:- type same_pair =< pair(int, int)
    --->    int - int.

:- type tagged(Phantom) =< colour
    --->    black.

:- type unit_tag ---> unit_tag.

main(!IO) :-
    W = red : warm,
    io.write_line(coerce(W) : bright, !IO),
    io.write_line(coerce(W) : colour, !IO),
    io.write_line(name(green), !IO),
    P = (3 - 4) : same_pair,
    io.write_line(coerce(P) : pair(int, int), !IO),
    K = black : tagged(unit_tag),
    io.write_line(coerce(K) : colour, !IO),
    N = make(5),
    io.write_line(coerce(N) : list(int), !IO).

:- func name(bright) = string.
name(red) = "red".
name(green) = "green".
name(blue) = "blue".

:- func make(T) = nel(T).
make(X) = [X, X].
