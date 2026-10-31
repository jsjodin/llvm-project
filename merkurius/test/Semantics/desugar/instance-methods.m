% An instance's method clauses are desugared as other clauses are, and a
% method defined by name is the clause that calls its implementation, or
% for a function unifies the result with it.
%
% RUN: merkc -fc1 -fdebug-dump-semantic-form %s | FileCheck %s

:- module instance_methods.
:- interface.
:- typeclass c(T) where [ func f(T) = T, pred p(T::in, T::out) is det,
    pred q(T::in, int::in, int::out) is det ].
:- implementation.
:- import_module int.
:- instance c(int) where [
    func(f/1) is g,
    pred(p/2) is r,
    (q(X) --> { X = 0 })
].
:- func g(int) = int.
g(X) = X.
:- pred r(int::in, int::out) is det.
r(X, X).
% CHECK:      method of instance c:
% CHECK-NEXT: Function f($0) = $1
% CHECK-NEXT: | Unify #0 $1 = g($0)
% CHECK-NEXT: method of instance c:
% CHECK-NEXT: Predicate p($0, $1)
% CHECK-NEXT: | Call #0 r($0, $1)
% CHECK-NEXT: method of instance c:
% CHECK-NEXT: Predicate q(X, dcg!0, dcg!1)
% CHECK-NEXT: | Conj #2
% CHECK-NEXT: | | Unify #0 X = 0
% CHECK-NEXT: | | Unify #1 dcg!1 = dcg!0
