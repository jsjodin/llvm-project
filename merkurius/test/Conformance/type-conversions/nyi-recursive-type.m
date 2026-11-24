% MANUAL: Type conversions
% A parameter of a recursive type of the exact form base(B1, ..., Bn) may
% differ: a tree(citrus) converts to a tree(fruit). A ground argument of a
% subtype gives a bound inst of all of its constructors, which is ground at
% the recursive node.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type nyi_recursive_type.tree(nyi_recursive_type.citrus)

:- module nyi_recursive_type.
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

:- type tree(T)
    --->    leaf
    ;       node(tree(T), T, tree(T)).

:- func up(tree(citrus)) = tree(fruit).
up(T) = coerce(T).

:- func count_lemons(tree(fruit)) = int.
count_lemons(leaf) = 0.
count_lemons(node(L, X, R)) =
    count_lemons(L) + (if X = lemon then 1 else 0) + count_lemons(R).

:- import_module int.

main(!IO) :-
    T = node(node(leaf, lemon, leaf), orange, node(leaf, lemon, leaf)),
    io.write_line(up(T), !IO),
    io.write_line(count_lemons(up(T)), !IO).
