% The constraints of undeclared predicates and functions are inferred with
% their types, as mmc does: a goal's constraints, reduced by the
% instances, that are on the head's type variables. A ground one is
% checked instead, and a superclass constraint is not dropped for its
% subclass's. Through calls, closures and recursion, they reach a
% fixpoint with the types.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module inference.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass loud(T) <= show(T) where [ func loud(T) = string ].
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- type w(T) ---> w(T).
:- instance show(int) where [ show(X) = string.int_to_string(X) ].
:- instance show(list(T)) <= show(T) where [ show(_) = "[...]" ].
:- instance loud(int) where [ loud(X) = show(X) ++ "!" ].
:- instance conv(int, string) where [ conv(X) = show(X) ].

main(0).
simple(X) = show(X).
% CHECK: clause at [[@LINE-1]]:1 of func inference.simple/1: inferred func inference.simple(T1) = string <= (inference.show(T1))
reduced(X) = show([[X]]).
% CHECK: clause at [[@LINE-1]]:1 of func inference.reduced/1: inferred func inference.reduced(T1) = string <= (inference.show(T1))
% CHECK: constraint of show at [[@LINE-2]]:14: inference.show(list.list(list.list(T1))) by the instance at 22:13
% CHECK-NEXT:   inference.show(list.list(T1)) by the instance at 22:13
% CHECK-NEXT:     inference.show(T1) by the declaration's inference.show(T1)
no_instance(X) = show(w(X)).
% CHECK: clause at [[@LINE-1]]:1 of func inference.no_instance/1: inferred func inference.no_instance(T1) = string <= (inference.show(inference.w(T1)))
both(X) = show(X) ++ loud(X).
% CHECK: clause at [[@LINE-1]]:1 of func inference.both/1: inferred func inference.both(T1) = string <= (inference.show(T1), inference.loud(T1))
converted(X) = Y :- Y = conv(X).
% CHECK: clause at [[@LINE-1]]:1 of func inference.converted/1: inferred func inference.converted(T1) = T2 <= (inference.conv(T1, T2))
ground = show(1) ++ loud(2) ++ conv(3).
% CHECK: clause at [[@LINE-1]]:1 of func inference.ground/0: inferred func inference.ground = string
two(X, Y) = show(X) ++ show([Y]) ++ show(X).
% CHECK: clause at [[@LINE-1]]:1 of func inference.two/2: inferred func inference.two(T1, T2) = string <= (inference.show(T1), inference.show(T2))
recursive([]) = "".
recursive([X | Xs]) = show(X) ++ recursive(Xs).
% CHECK: clause at [[@LINE-1]]:1 of func inference.recursive/1: inferred func inference.recursive(list.list(T1)) = string <= (inference.show(T1))
mutual(X) = show(X) ++ mutual2(X).
mutual2(X) = loud(X) :- _ = mutual(X).
% CHECK: clause at [[@LINE-2]]:1 of func inference.mutual/1: inferred func inference.mutual(T1) = string <= (inference.show(T1), inference.loud(T1))
% CHECK: clause at [[@LINE-2]]:1 of func inference.mutual2/1: inferred func inference.mutual2(T1) = string <= (inference.loud(T1), inference.show(T1))
caller(X) = simple(X) ++ simple(3) ++ "".  % "" decides mmc's first pass
% CHECK: clause at [[@LINE-1]]:1 of func inference.caller/1: inferred func inference.caller(T1) = string <= (inference.show(T1))
% CHECK: constraint of simple at [[@LINE-2]]:13: inference.show(T1) by the declaration's inference.show(T1)
% CHECK: constraint of simple at [[@LINE-3]]:26: inference.show(int) by the instance at 21:13
closure = list.map(show).
% CHECK: clause at [[@LINE-1]]:1 of func inference.closure/0: inferred func inference.closure = (func(list.list(T1)) = list.list(string)) <= (inference.show(T1))
:- func declared(T) = string <= loud(T).
declared(X) = both(X).
% CHECK: constraint of both at [[@LINE-1]]:15: inference.show(T) by the declaration's inference.loud(T)
% CHECK: constraint of both at [[@LINE-2]]:15: inference.loud(T) by the declaration's inference.loud(T)
% An assignment whose constraints no instance satisfies is dropped: of
% string.length's function and the closure of its predicate, only int
% has an instance of show.
length_shown(S) = show(string.length(S)).
% CHECK: clause at [[@LINE-1]]:1 of func inference.length_shown/1: inferred func inference.length_shown(string) = string
% So is a combination of clauses' typings: the second clause's constraint
% decides the first's overloading.
:- func any_shown(float) = T <= show(T).
any_shown(X) = any_shown(X).
combined(S) = string.length(S).
combined(_) = any_shown(1.5).
% CHECK: clause at [[@LINE-2]]:1 of func inference.combined/1: inferred func inference.combined(string) = int
