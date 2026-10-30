% Undeclared predicates and functions whose clauses fit more than one
% type, none more general than the others, are ambiguous (reference
% manual, "Predicate and function type declarations"), as mmc reports too.
% Most are a function whose result is a call of a name that is both a
% function and a predicate, which fits the function's result or a closure
% of the predicate. In item and length, two typings each make a fixpoint.
% Each is checked without a signature, and reports its ambiguity as a
% declared one's would.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --allow-errors %s %}

:- module ambiguity.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, pair, string.

:- type item ---> length(int) ; append.
:- type either(L, R) ---> left(L) ; right(R).

main(0).
library_append(Xs) = list.append(Xs, [1]).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:22: because: list.append may be pred list.append/3 or func list.append/2
length(item) = 1.
item = length(3).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:8: because: length may be constructor ambiguity.length/1 or func ambiguity.length/1
filter_pos(Xs) = list.filter(positive, Xs).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:18: because: list.filter may be func list.filter/2 or pred list.filter/3 or pred list.filter/4
positive(X) :- X > 0.
string_length(S) = string.length(S) :- S = "abc".
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:20: because: string.length may be func string.length/1 or pred string.length/2
split(left(L), L).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:7: because: left may be constructor ambiguity.left/1 or func string.left/2 or pred string.left/3
first_of(N - _) = N.
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:12: because: - may be func int.'-'/2 or constructor pair.'-'/2
names(Ns) = map(fst, Ns).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:17: because: fst may be func pair.fst/1 or pred pair.fst/2
% CHECK: ambiguity.m:[[@LINE-3]]:13: because: map may be func list.map/2 or pred list.map/3
own_append = append.
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:14: because: append may be constructor ambiguity.append/0 or pred list.append/3 or func list.append/2 or func string.append/2 or pred string.append/3
min_of(X, Y) = int.min(X, Y).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:16: because: int.min may be func int.min/2 or pred int.min/3
% A caller of an ambiguous function is checked without its signature.
uses_ambiguous = library_append([2]).
