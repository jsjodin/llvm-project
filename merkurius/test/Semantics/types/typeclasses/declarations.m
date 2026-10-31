% Type classes and instances that follow the rules: superclasses, also on
% ground types; instances with constraints, abstract ones with their
% definitions; methods defined by clauses and by name; and signatures with
% constraints, which a method's has from its type class.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module declarations.
:- interface.
:- import_module list.
:- typeclass shape(T) where [
    func area(T) = float,
    pred scale(float::in, T::in, T::out) is det
].
:- typeclass named(T) <= shape(T) where [ func name(T) = string ].
:- typeclass convert(A, B) where [ func convert(A) = B ].
:- typeclass marker(T).
:- type square ---> square(float).
:- instance shape(square).
:- instance named(square).
:- instance convert(square, string).
:- pred describe(T::in, string::out) is det <= named(T).
:- implementation.
:- import_module float, string.

:- typeclass marker(T) where [].
:- typeclass counted(T) <= (shape(T), marker(int)) where [].
:- type circle ---> circle(float).
:- instance marker(int) where [].
:- instance shape(square) where [
    area(square(S)) = S * S,
    scale(F, square(S), square(F * S))
].
:- instance named(square) where [ func(name/1) is square_name ].
:- instance shape(circle) where [
    func(area/1) is circle_area,
    pred(scale/3) is scale_circle
].
:- instance convert(square, string) where [
    convert(square(S)) = string.float_to_string(S)
].
:- instance convert({A, B}, list(B)) where [ convert({_, Y}) = [Y] ].
:- instance shape(list(T)) <= shape(T) where [
    area(Xs) = sum_areas(Xs),
    scale(F, Xs, map(scaled(F), Xs))
].

describe(X, name(X) ++ ": " ++ string.float_to_string(area(X))).
% CHECK: clause at [[@LINE-1]]:1 of pred declarations.describe/2: pred declarations.describe(T, string) <= (declarations.named(T))

:- func square_name(square) = string.
square_name(_) = "square".
:- func circle_area(circle) = float.
circle_area(circle(R)) = 3.0 * R * R.
:- pred scale_circle(float::in, circle::in, circle::out) is det.
scale_circle(F, circle(R), circle(F * R)).
:- func sum_areas(list(T)) = float <= shape(T).
sum_areas([]) = 0.0.
sum_areas([X | Xs]) = area(X) + sum_areas(Xs).
% CHECK: clause at [[@LINE-2]]:1 of func declarations.sum_areas/1: func declarations.sum_areas(list.list(T)) = float <= (declarations.shape(T))
:- func scaled(float, T) = T <= shape(T).
scaled(F, X) = Y :- scale(F, X, Y).
:- func converted_area(A, B) = float <= (convert(A, B), shape(B)).
converted_area(X, Y) = area(Y) :- Y = convert(X).
% CHECK: clause at [[@LINE-1]]:1 of func declarations.converted_area/2: func declarations.converted_area(A, B) = float <= (declarations.convert(A, B), declarations.shape(B))
