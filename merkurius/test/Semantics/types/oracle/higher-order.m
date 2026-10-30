% Higher-order code: closures of predicates, functions and constructors,
% partial application, lambda expressions, higher-order calls with call/N
% and apply/N, and functions that take and return functions.
%
% REQUIRES: mmc
% RUN: %compare-types %s

:- module higher_order.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string.

main(0).
my_map(_, []) = [].
my_map(F, [X | Xs]) = [F(X) | my_map(F, Xs)].
my_map_pred(_, [], []).
my_map_pred(P, [X | Xs], [Y | Ys]) :- P(X, Y), my_map_pred(P, Xs, Ys).
my_foldl(_, [], A, A).
my_foldl(P, [X | Xs], A0, A) :- P(X, A0, A1), my_foldl(P, Xs, A1, A).
my_foldr(_, [], A) = A.
my_foldr(F, [X | Xs], A) = F(X, my_foldr(F, Xs, A)).
filter_pos(Xs) = list.filter(func_pos, Xs) ++ [].
func_pos(X) :- X > 0.
apply_twice(F, X) = F(F(X)).
apply_with_apply(F, X) = apply(F, X).
call_with_call(P, X) :- call(P, X).
call_with_extra(P, X, Y) :- call(P, X, Y).
compose(F, G) = (func(X) = F(G(X))).
curry(F) = (func(X) = (func(Y) = F(X, Y))).
uncurry(F) = (func(X, Y) = apply(F(X), Y)).
flip(P) = (pred(X::in, Y::in) is semidet :- P(Y, X)).
constant(K) = (func(_) = R :- R = K).
identity = (func(X) = X).
add(N) = (func(X) = X + N).
partial_pred(Xs, Ys) :- my_map_pred(int_double, Xs, Ys).
int_double(X, Y) :- Y = X * 2.
partial_func(Xs) = my_map(int.plus(1), Xs).
lambda_pred(Xs) = Ys :- list.filter((pred(X::in) is semidet :- X > 1), Xs, Ys).
nested(Xss) = my_map(func(Xs) = my_map(func(X) = X + 1, Xs), Xss).
returns_pred = P :- P = (pred(X::out) is det :- X = "a").
pipeline(X) = apply_twice(add(1), apply_twice(compose(add(2), add(3)), X)).
sum_by(F, Xs) = my_foldr(func(X, A) = F(X) + A, Xs, 0).
closure_of_closure(P) = (pred(X::in) is semidet :- call(P, X)).
pred_to_func(P) = (func(X) = Y :- P(X, Y)).
higher_order_tuple = {add(1), my_map(add(2))}.
