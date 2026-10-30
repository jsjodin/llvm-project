% Constructors of the module's own types and the library's: enums,
% polymorphic types, recursive types, constructors overloaded between
% types (red), and equivalence types. An undeclared function whose result
% is a call of a name that is both a function and a predicate, such as
% map, fits two types, so such a result is used in a call of ++.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module constructors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, maybe, pair.

:- type colour ---> red ; green ; blue.
:- type tree(T) ---> leaf ; node(tree(T), T, tree(T)).
:- type shape ---> circle(float) ; rect(float, float).
:- type either(L, R) ---> left(L) ; right(R).
% red also names a constructor of this type; the context picks one.
:- type light ---> red ; amber ; off.
:- type wrapper ---> wrap(colour).
:- type name == string.
:- type named(T) == pair(name, T).
:- type phantom(T) ---> phantom.

main(0).
a_colour = green.
colours = [red, green, blue].
a_light = amber.
red_light(L) :- L = red, L = off.
red_colour(C) :- C = red, C = blue.
wrapped_red = wrap(red).
a_tree = node(leaf, 1, node(leaf, 2, leaf)).
tree_insert(X, leaf) = node(leaf, X, leaf).
tree_insert(X, node(L, Y, R)) = node(L, Y, tree_insert(X, R)).
tree_size(leaf) = 0.
tree_size(node(L, _, R)) = tree_size(L) + 1 + tree_size(R).
tree_flatten(leaf) = [].
tree_flatten(node(L, X, R)) = tree_flatten(L) ++ [X | tree_flatten(R)].
area(circle(R)) = 3.0 * R * R.
area(rect(W, H)) = W * H.
choose(yes(X), _) = X.
choose(no, D) = D.
split(left(L), L, R) :- R = "none".
split(right(R), "none", R).
a_pair = 1 - "one".
named_value(V) = "n" - V.
names(Ns) = map(fst, Ns) ++ [].
:- func first_of(pair(T, U)) = T.
first_of(N - _) = N.
a_phantom = phantom.
phantom_int(P) :- P = phantom, _ = P : phantom(int).
maybe_head([]) = no.
maybe_head([X | _]) = yes(X).
wrap_all(Cs) = map(func(C) = wrap(C), Cs).
