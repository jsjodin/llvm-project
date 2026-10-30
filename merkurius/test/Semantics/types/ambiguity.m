% A clause must have one type assignment, up to renaming: two that chose
% different declarations for a name are an ambiguity, which mmc reports in
% the same words. A type that nothing determines draws mmc's warning.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck %s

:- module ambiguity.
:- interface.
:- import_module list.
:- func sum(list(int)) = int.
:- pred p is det.
:- implementation.
:- import_module int.

:- func plus(int, int) = int.
plus(X, Y) = X + Y.

sum(Xs) = foldl(plus, Xs, 0).
% CHECK: ambiguity.m:[[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: ambiguity.m:[[@LINE-2]]:17: because: plus may be func ambiguity.plus/2 or func int.plus/2

p :- X = [], X = X.
% CHECK: ambiguity.m:[[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK-NEXT: X: list.list(T_{{[0-9]+}})
