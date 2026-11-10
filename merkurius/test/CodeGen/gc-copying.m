% RUN: merkc -O0 %s -o %t0
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t0.O2
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t
% RUN: env MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=7 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% A copying collection at every allocation must not change what a program
% prints: every kind of value is live across allocations here.
:- module gc_copying.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, int, list, string.

:- type tree(T)
    --->    leaf
    ;       node(tree(T), T, tree(T)).

:- type shape
    --->    circle(string, int)
    ;       square(int)
    ;       point.

:- typeclass describe(T) where [
    func describe(T) = string
].
:- instance describe(int) where [
    func(describe/1) is string.int_to_string
].
:- instance describe(shape) where [
    func(describe/1) is shape_name
].

:- type box
    --->    some [T] box(T) => describe(T).

:- func shape_name(shape) = string.
shape_name(circle(Name, R)) = Name ++ string.int_to_string(R).
shape_name(square(S)) = "square" ++ string.int_to_string(S).
shape_name(point) = "point".

:- func insert(T, tree(T)) = tree(T).
insert(X, leaf) = node(leaf, X, leaf).
insert(X, node(L, Y, R)) = Result :-
    compare(Order, X, Y),
    (
        Order = (<),
        Result = node(insert(X, L), Y, R)
    ;
        Order = (>),
        Result = node(L, Y, insert(X, R))
    ;
        Order = (=),
        Result = node(L, X, R)
    ).

:- func flatten(tree(T)) = list(T).
flatten(leaf) = [].
flatten(node(L, X, R)) = flatten(L) ++ [X | flatten(R)].

:- func lists(array(list(int))) = list(list(int)).
lists(Array) = [array.lookup(Array, 0), array.lookup(Array, 1),
    array.lookup(Array, 2), array.lookup(Array, 3), array.lookup(Array, 4)].

:- func describe_all(list(box)) = list(string).
describe_all([]) = [].
describe_all([box(X) | Xs]) = [describe(X) | describe_all(Xs)].

main(!IO) :-
    Numbers = 1 `..` 40,
    Tree = list.foldl(insert, reverse(Numbers), leaf),
    io.write_int(list.foldl(plus, flatten(Tree), 0), !IO),
    io.nl(!IO),
    Offset = 3,
    AddOffset = (func(X) = X + Offset),
    io.write(list.map(AddOffset, 1 `..` 5), !IO),
    io.nl(!IO),
    Names = list.map(func(I) = "n" ++ string.int_to_string(I), 1 `..` 6),
    io.write_string(string.join_list(",", Names), !IO),
    io.nl(!IO),
    Shapes = [circle("c", 1), square(2), point],
    io.write(flatten(list.foldl(insert, Shapes, leaf)), !IO),
    io.nl(!IO),
    Boxes = ['new box'(1), 'new box'(square(7)), 'new box'(circle("r", 2))],
    io.write(describe_all(Boxes), !IO),
    io.nl(!IO),
    array.from_list(list.map(func(I) = [I, I * I], 1 `..` 5), Array),
    io.write(lists(Array), !IO),
    io.nl(!IO),
    Pair = {"left", [1.5, 2.5]},
    Longer = list.map(func(I) = I + 1, 1 `..` 30),
    io.write(Pair, !IO),
    io.nl(!IO),
    io.write_int(list.length(Longer), !IO),
    io.nl(!IO).

% OUTPUT:820
% OUTPUT-NEXT:[4, 5, 6, 7, 8]
% OUTPUT-NEXT:n1,n2,n3,n4,n5,n6
% OUTPUT-NEXT:[circle("c", 1), square(2), point]
% OUTPUT-NEXT:["1", "square7", "r2"]
% OUTPUT-NEXT:{{\[\[}}1, 1], [2, 4], [3, 9], [4, 16], [5, 25]]
% OUTPUT-NEXT:{"left", [1.5, 2.5]}
% OUTPUT-NEXT:30
