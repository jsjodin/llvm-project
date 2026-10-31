% The rules for functional dependencies on instances and signatures
% (reference manual, "Functional dependencies"): two instances whose
% domains unify have the same range, an instance's domain determines the
% type variables of its range, with its constraints, and without a
% dependency a constraint's variables must occur in the signature.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module dependency_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list, pair, string.

main(0).
:- typeclass baz(A, B) <= (A -> B) where [ func baz(A) = B ].
:- typeclass nz(A, B) where [ func nz(A) = B ].

:- instance baz(int, int) where [ baz(X) = X ].
:- instance baz(int, string) where [ baz(_) = "" ].
% CHECK: [[@LINE-1]]:13: error: instance dependency_errors.baz(int, string) is inconsistent with an earlier instance by the functional dependency (A -> B) of typeclass baz/2
% CHECK: [[@LINE-3]]:13: because: the earlier instance is dependency_errors.baz(int, int)
% Domains with variables unify too.
:- instance baz(pair(S, T), int) where [ baz(_) = 0 ].
:- instance baz(pair(U, V), string) where [ baz(_) = "" ].
% CHECK: [[@LINE-1]]:13: error: instance dependency_errors.baz(pair.pair(U, V), string) is inconsistent with an earlier instance by the functional dependency (A -> B) of typeclass baz/2
% CHECK: [[@LINE-3]]:13: because: the earlier instance is dependency_errors.baz(pair.pair(S, T), int)
:- instance baz(string, list(T)) where [ baz(_) = [] ].
% CHECK: [[@LINE-1]]:13: error: in instance dependency_errors.baz(string, list.list(T)), type variable T is in the range of the functional dependency (A -> B), but its domain does not determine it
% The instance's constraint determines T from S.
:- instance baz(list(S), list(T)) <= baz(S, T) where [ baz(_) = [] ].

% With several dependencies, each must hold, and a range of two types.
:- typeclass dep(A, B, C) <= ((A -> B, C), (C -> A)) where [].
:- instance dep(list(S), pair(S, T), int) <= baz(S, T) where [].
% CHECK: [[@LINE-1]]:13: error: in instance dependency_errors.dep(list.list(S), pair.pair(S, T), int), type variable S is in the range of the functional dependency (C -> A), but its domain does not determine it
:- instance dep(int, pair(S, T), list(S)) where [].
% CHECK: [[@LINE-1]]:13: error: in instance dependency_errors.dep(int, pair.pair(S, T), list.list(S)), type variable S is in the range of the functional dependency (A -> B, C), but its domain does not determine it
% CHECK: [[@LINE-2]]:13: error: in instance dependency_errors.dep(int, pair.pair(S, T), list.list(S)), type variable T is in the range of the functional dependency (A -> B, C), but its domain does not determine it
:- instance dep(string, string, list(T)) where [].
% CHECK: [[@LINE-1]]:13: error: in instance dependency_errors.dep(string, string, list.list(T)), type variable T is in the range of the functional dependency (A -> B, C), but its domain does not determine it
% CHECK: [[@LINE-2]]:13: error: instance dependency_errors.dep(string, string, list.list(T)) is inconsistent with an earlier instance by the functional dependency (C -> A) of typeclass dep/3
% CHECK: [[@LINE-6]]:13: because: the earlier instance is dependency_errors.dep(int, pair.pair(S, T), list.list(S))

:- pred p(X::in) is det <= nz(X, Z).
% CHECK: [[@LINE-1]]:4: error: type variable Z occurs in the constraints of predicate p/1, but not in its argument types
p(_).
