% The types of predicates and functions without declarations are inferred,
% as mmc's --infer-types does, and this module passes
% mmc --errorcheck-only --infer-all, which infers the same signatures, up
% to the names of their type variables. Inference takes the call graph's
% strongly connected components callees first, such as evens and odds
% together; each starts most general and is recomputed from its clauses
% until it does not change. Callers then use the inferred signature, as
% they do a declared one, so twice's result is an int in main_len.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --implicit-check-not=warning %s

:- module inference.
:- interface.
:- import_module list.
:- pred main_len(list(int)::in, int::out) is det.
:- implementation.
:- import_module int.

main_len(L, N) :- len(L, N0), N = twice(N0), _ = evens([1, 2]), swap(1, "a", _, _).
len([], 0).
len([_ | T], N) :- len(T, N0), N = N0 + 1.
twice(X) = X + X.
evens([]) = [].
evens([X | Xs]) = [X | odds(Xs)].
odds([]) = [].
odds([_ | Xs]) = evens(Xs).
swap(A, B, B, A).
app(F, X) = F(X).
compose(F, G, X) = app(F, app(G, X)).
grow(X, Y) :- grow([X], Y).

% CHECK:      clause at 19:1 of pred inference.main_len/2: pred inference.main_len(list.list(int), int)
% CHECK-NEXT:   L: list.list(int)
% CHECK-NEXT:   N: int
% CHECK-NEXT:   N0: int
% CHECK:      clause at 20:1 of pred inference.len/2: inferred pred inference.len(list.list(T1), int)
% CHECK:      clause at 22:1 of func inference.twice/1: inferred func inference.twice(int) = int
% CHECK:      clause at 23:1 of func inference.evens/1: inferred func inference.evens(list.list(T1)) = list.list(T1)
% CHECK:      clause at 25:1 of func inference.odds/1: inferred func inference.odds(list.list(T1)) = list.list(T1)
% CHECK:      clause at 27:1 of pred inference.swap/4: inferred pred inference.swap(T1, T2, T2, T1)
% CHECK:      clause at 28:1 of func inference.app/2: inferred func inference.app((func(T1) = T2), T1) = T2
% CHECK:      clause at 29:1 of func inference.compose/3: inferred func inference.compose((func(T1) = T2), (func(T3) = T1), T3) = T2
% CHECK:      clause at 30:1 of pred inference.grow/2: inferred pred inference.grow(T1, T2)
