% The shapes of type inference: recursion, mutual recursion, polymorphism,
% inferred predicates calling each other and declared ones, declared
% predicates calling inferred ones, recursion at other types (which the
% passes allow, as declarations do), a predicate whose clauses together
% are less general than each one, and unused arguments.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module inference.
:- interface.
:- import_module list.
:- pred main(int::out) is det.
:- func declared_caller(list(int)) = int.
:- implementation.
:- import_module int, string.

main(0).
declared_caller(Xs) = length_of(Xs) + sum_of(Xs).
length_of([]) = 0.
length_of([_ | Xs]) = 1 + length_of(Xs).
sum_of([]) = 0.
sum_of([X | Xs]) = X + sum_of(Xs).
even_len([]).
even_len([_ | Xs]) :- odd_len(Xs).
odd_len([_ | Xs]) :- even_len(Xs).
ping(N, S) :- ( if N = 0 then S = "" else pong(N - 1, S0), S = "i" ++ S0 ).
pong(N, S) :- ( if N = 0 then S = "" else ping(N - 1, S0), S = "o" ++ S0 ).
identity(X) = X.
uses_identity = {identity(1), identity("a")}.
first(X, _) = X.
second(_, Y) = Y.
same(X, X).
unused(_, _, _).
clauses_combine(1, _).
clauses_combine(_, "a").
reverse_acc([], Acc) = Acc.
reverse_acc([X | Xs], Acc) = reverse_acc(Xs, [X | Acc]).
calls_declared(Xs) = list.length(Xs) + declared_caller(Xs).
nested_pairs(X) = {X, {X, X}}.
recursion_at_other_type(X, N) :- ( if N = 0 then true else recursion_at_other_type({X}, N - 1) ).
three_cycle_a(X) :- three_cycle_b(X).
three_cycle_b(X) :- three_cycle_c(X).
three_cycle_c(X) :- X = [], three_cycle_a(X).
callee_later(X) = callee_defined_after(X) + 1.
callee_defined_after(X) = X * 2.
zero_arity_pred :- true.
zero_arity_func = "z".
uses_zero = zero_arity_func ++ "!".
