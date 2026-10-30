% Overloading, which the types of arguments and results settle: names
% defined both here and in the library, functions and predicates of the
% same name, operators on several types, and constructors that are also
% functions. A function's result that would otherwise fit both a function
% and a closure of a predicate is used in a call of + or ++.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module overloading.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- type item ---> length(int) ; append.

main(0).
% The constructor length/1, list's and string's length: the argument
% settles which.
list_length(Xs) = list.length(Xs) + 0 :- Xs = [1].
string_length(S) = string.length(S) + 0 :- S = "abc".
by_argument_type = list.length([1, 2]) + string.length("ab").
own_append(X) :- X = append, X = length(1).
library_append(Xs) = list.append(Xs, [1]) ++ [].
int_plus(X) = X + 1.
float_plus(X) = X + 1.0.
string_plus(X) = X ++ "!".
list_plus(X) = X ++ [1].
mixed = {1 + 2, 1.0 + 2.0, "a" ++ "b", [1] ++ [2]}.
pred_and_func(Xs, N) :- list.length(Xs, N), N = list.length(Xs).
min_of(X, Y) = int.min(X, Y) + 0.
