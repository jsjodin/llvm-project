% A constraint whose domain no instance matches is not improved, so
% nothing determines its range. Inference does not keep a constraint on a
% type variable that the head does not have, even one a functional
% dependency would determine from the head's, as mmc does not (see the
% register): with `:- func hidden(T) = int <= baz(T, U)` declared, hidden
% would be valid.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module improvement_undetermined.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, string.

main(0).
:- typeclass baz(A, B) <= (A -> B) where [ func baz(A) = B ].
:- typeclass sh(T) where [ func sh(T) = string ].
:- instance baz(int, string) where [ baz(X) = string.int_to_string(X) ].

no_instance = 0 :- _ = baz(1.5).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:24: error: unsatisfied type class constraint `improvement_undetermined.baz(float, T_4)': nothing determines its type
hidden(X) = 0 :- _ = baz(X).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:22: error: unsatisfied type class constraint `improvement_undetermined.baz(T1, T_5)': nothing determines its type
hidden_shown(X) = sh(baz(X)).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:19: error: unsatisfied type class constraint `improvement_undetermined.sh(T_6)': nothing determines its type
% CHECK: [[@LINE-3]]:22: error: unsatisfied type class constraint `improvement_undetermined.baz(T1, T_6)': nothing determines its type
