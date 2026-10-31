% Functional dependencies determine types (reference manual, "Functional
% dependencies"): a constraint's range is that of an instance whose types
% match its domain, with the instance's constraints improving in turn, or
% that of another constraint of the class with its domain, wanted or
% assumed. Signatures may constrain type variables that dependencies
% determine, and a type class's methods need not use them. Improvement
% comes after overloading is resolved, as in mmc.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module improvement.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, pair, string.

main(0).
:- typeclass baz(A, B) <= (A -> B) where [ func baz(A) = B ].
:- typeclass sh(T) where [ func sh(T) = string ].
:- instance baz(int, string) where [ baz(X) = string.int_to_string(X) ].
:- instance baz(list(S), list(T)) <= baz(S, T) where [ baz(_) = [] ].
:- instance sh(string) where [ sh(S) = S ].

ground = baz(1).
% CHECK: clause at [[@LINE-1]]:1 of func improvement.ground/0: inferred func improvement.ground = string
% CHECK: constraint of baz at [[@LINE-2]]:10: improvement.baz(int, string) by the instance at 21:13
in_list = baz([1]).
% CHECK: clause at [[@LINE-1]]:1 of func improvement.in_list/0: inferred func improvement.in_list = list.list(string)
% CHECK: constraint of baz at [[@LINE-2]]:11: improvement.baz(list.list(int), list.list(string)) by the instance at 22:13
% CHECK-NEXT: improvement.baz(int, string) by the instance at 21:13
shown = sh(baz(2)).
% CHECK: clause at [[@LINE-1]]:1 of func improvement.shown/0: inferred func improvement.shown = string
general(X) = baz(X).
% CHECK: clause at [[@LINE-1]]:1 of func improvement.general/1: inferred func improvement.general(T1) = T2 <= (improvement.baz(T1, T2))
twice(X) = {baz(X), baz(X)}.
% CHECK: clause at [[@LINE-1]]:1 of func improvement.twice/1: inferred func improvement.twice(T1) = {T2, T2} <= (improvement.baz(T1, T2))
by_general = general(3) ++ "".
% CHECK: clause at [[@LINE-1]]:1 of func improvement.by_general/0: inferred func improvement.by_general = string

% The declaration's constraint determines the range of the goal's.
:- pred ranged(X::in, Y::out) is det <= baz(X, Y).
ranged(X, Y) :- Y = baz(X).
:- pred hidden_range(X::in) is det <= (baz(X, Y), sh(Y)).
hidden_range(X) :- Z = baz(X), ranged(X, Z), _ = sh(Z).
% CHECK: clause at [[@LINE-1]]:1 of pred improvement.hidden_range/1: pred improvement.hidden_range(X) <= (improvement.baz(X, Y), improvement.sh(Y))
% CHECK: Z: Y

% Z is determined through the domain pair(X, Y), and P through quux's
% superclass baz(R, P), as in the manual's example; quux's method need not
% use P.
:- pred through(X::in, Y::in) is det <= baz(pair(X, Y), list(Z)).
through(_, _).
:- typeclass quux(P, Q, R) <= baz(R, P) where [ func quux(Q, R) = int ].
:- pred super(Q::in, R::in) is det <= quux(P, Q, R).
super(_, _).
:- typeclass one(A, B) <= ((A -> B), (B -> A)) where [ func one(A) = int ].
