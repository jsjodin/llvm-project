% What target regions cannot do: map a value that can never be flat on a
% device, a string, a closure, a term of a recursive type such as a list,
% or anything that holds one, which is an error that names the variable
% and its type (design document, "Target regions"). And what they cannot
% do yet: map other values that are not scalars, arrays of them or terms
% whose cells hold only scalars, give terms, or take clauses; and the other
% directives. A region's goal must be det, since a region cannot fail.
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:

:- module omp_target_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- pred same(string::in, string::out) is det.
same(S, T) :-
    %$omp target
    T = S.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: T has type string, which cannot be mapped to a device

:- pred first(array(string)::array_ui, string::out) is det.
first(A, S) :-
    %$omp target
    S = A ^ elem(0).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: A has type array.array(string), which cannot be mapped to a device

:- type shape
    --->    circle(float, float)
    ;       square(float, float).

:- pred width(shape::in, float::out) is det.
width(Shape, W) :-
    %$omp target
    (
        Shape = circle(W, _)
    ;
        Shape = square(W, _)
    ).
% CHECK: omp-target-errors.m:[[@LINE-6]]:5: error: not yet implemented: code generation for values of type omp_target_errors.shape in target regions

:- type named
    --->    named(string, int).

:- pred number(named::in, int::out) is det.
number(Named, N) :-
    %$omp target
    Named = named(_, N).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: Named has type omp_target_errors.named, which cannot be mapped to a device

:- type pair
    --->    pair(int, int).

:- pred double(int::in, pair::out) is det.
double(X, P) :-
    %$omp target
    P = pair(X, X).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for values of type omp_target_errors.pair in target regions

:- pred total(list(int)::in, int::out) is det.
total(Xs, S) :-
    %$omp target
    S = list.foldl(int.plus, Xs, 0).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: Xs has type list.list(int), which cannot be mapped to a device

:- type tree
    --->    leaf
    ;       node(tree, int, tree).

:- pred root(tree::in, int::out) is det.
root(T, R) :-
    %$omp target
    ( T = node(_, R, _) ; T = leaf, R = 0 ).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: T has type omp_target_errors.tree, which cannot be mapped to a device

:- pred apply(func(int) = int::in, int::in, int::out) is det.
apply(F, X, Y) :-
    %$omp target
    Y = F(X).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: F has type (func(int) = int), which cannot be mapped to a device

:- pred positive(int::in, int::out) is semidet.
positive(X, Y) :-
    %$omp target
    ( X > 0, Y = X ).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: the goal of a target directive must be det, but it is semidet

:- pred halve(int::in, int::out) is semidet.
halve(X, Y) :-
    %$omp target teams
    ( X mod 2 = 0, Y = X / 2 ).
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: the goal of a target teams directive must be det, but it is semidet

:- pred next(int::in, int::out) is det.
next(X, Y) :-
    %$omp target map(tofrom: X)
    Y = X + 1.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the map clause of the target directive

:- pred twice(int::in, int::out) is det.
twice(X, Y) :-
    %$omp parallel
    Y = X * 2.
% CHECK: omp-target-errors.m:[[@LINE-2]]:5: error: not yet implemented: code generation for the parallel directive

main(!IO) :-
    same("abc", _),
    N = 3,
    first(array.init(1, "a"), _),
    width(circle(1.0, 2.0), _),
    number(named("a", 1), _),
    double(1, _),
    total([1, 2], _),
    root(node(leaf, 1, leaf), _),
    apply(int.plus(1), 1, _),
    ( if positive(1, _), halve(2, _) then true else true ),
    next(N, M),
    twice(M, K),
    io.write_int(K, !IO).
