% Inferring the types of higher-order code with no declarations: closures,
% calls of higher-order arguments, partial application, and functions that
% return functions. These are the most general valid types, which mmc
% --infer-all also infers without odd's error. With it, mmc stops after its
% first pass, reports lengths as ambiguous, and prints first guesses for
% sum and twice (see the design document's register).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck %s

:- module higher_order_inference.
:- interface.
:- pred main(int::out) is det.
:- pred odd(string::out) is det.
:- implementation.
:- import_module int, list, string.

main(N) :- N = sum(mymap(add(1), myfilter(positive, [-1, 2, 3]))).
mymap(_, []) = [].
mymap(F, [X | Xs]) = [F(X) | mymap(F, Xs)].
myfilter(_, []) = [].
myfilter(P, [X | Xs]) = Ys :-
    ( if P(X) then Ys = [X | myfilter(P, Xs)] else Ys = myfilter(P, Xs) ).
myfoldl(_, [], A) = A.
myfoldl(F, [X | Xs], A) = myfoldl(F, Xs, F(X, A)).
add(X, Y) = X + Y.
positive(X) :- X > 0.
sum(Xs) = myfoldl(add, Xs, 0).
compose(F, G) = (func(X) = F(G(X))).
flip(F) = (func(A, B) = F(B, A)).
twice(F) = compose(F, F).
lengths(Ss) = mymap(string.length, Ss).
apply_pred(P, X, Y) :- P(X, Y).
odd(S) :- S = sum("a").
% CHECK: higher-order-inference.m:[[@LINE-1]]:15: error: type error in unification of variable S and `sum("a")': variable S has type string, which no declaration of sum/1 fits

% CHECK-DAG: inferred func higher_order_inference.mymap((func(T1) = T2), list.list(T1)) = list.list(T2)
% CHECK-DAG: inferred func higher_order_inference.myfilter(pred(T1), list.list(T1)) = list.list(T1)
% CHECK-DAG: inferred func higher_order_inference.myfoldl((func(T1, T2) = T2), list.list(T1), T2) = T2
% CHECK-DAG: inferred func higher_order_inference.sum(list.list(int)) = int
% CHECK-DAG: inferred func higher_order_inference.compose((func(T1) = T2), (func(T3) = T1)) = (func(T3) = T2)
% CHECK-DAG: inferred func higher_order_inference.flip((func(T1, T2) = T3)) = (func(T2, T1) = T3)
% CHECK-DAG: inferred func higher_order_inference.twice((func(T1) = T1)) = (func(T1) = T1)
% CHECK-DAG: inferred func higher_order_inference.lengths(list.list(string)) = list.list(int)
% CHECK-DAG: inferred pred higher_order_inference.apply_pred(pred(T1, T2), T1, T2)
