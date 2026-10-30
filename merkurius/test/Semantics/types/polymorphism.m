% The head must match the declaration exactly, up to renaming, so the
% declaration's type variables are not bound in a clause. A type
% qualification's variable with the name of one of them is that one; any
% other is the clause's own, the same in all its qualifications. mmc
% reports the same two errors.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck %s

:- module polymorphism.
:- interface.
:- pred id(T::in, T::out) is det.
:- pred one(T::in, int::out) is det.
:- pred two(T::in, T::out) is det.
:- implementation.

id(X, Y) :- Y = (X : T), Z = (X : T), Z = Z.
one(X, Y) :- Y = X.
% CHECK: polymorphism.m:[[@LINE-1]]:14: error: type error in unification of variable Y and variable X: variable Y has type int, but variable X has type T
two(X, Y) :- Y = (X : U), Y = (3 : U).
% CHECK: polymorphism.m:[[@LINE-1]]:32: error: type error in unification of variable Y and `3 : U': variable Y has type T, but `3 : U' has type int

% CHECK:      clause at 16:1 of pred polymorphism.id/2: pred polymorphism.id(T, T)
% CHECK-NEXT:   X: T
% CHECK-NEXT:   Y: T
% CHECK-NEXT:   Z: T
