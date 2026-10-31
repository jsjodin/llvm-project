% A declaration's type variables are each a type of its own, which the
% clauses cannot bind (reference manual, "Semantics of type quantifiers"):
% two of them do not unify, directly or as parts of other types.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module rigid_variables.
:- interface.
:- import_module list.
:- pred same(T::in, U::in) is semidet.
:- pred in_lists(list(T)::in, list(U)::in) is semidet.
:- pred same_twice(T::in, T::in) is semidet.
:- implementation.
same(X, Y) :- X = Y.
% CHECK: [[@LINE-1]]:15: error: type error in unification of variable X and variable Y: variable X has type T, but variable Y has type U
in_lists(Xs, Ys) :- Xs = Ys.
% CHECK: [[@LINE-1]]:21: error: type error in unification of variable Xs and variable Ys: variable Xs has type list.list(T), but variable Ys has type list.list(U)
same_twice(X, Y) :- X = Y.
