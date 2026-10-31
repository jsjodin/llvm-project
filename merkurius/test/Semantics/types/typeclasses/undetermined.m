% Constraints on a type that nothing determines, also after an instance
% reduces them. mmc reports these only in a module without unsatisfiable
% constraints, so they have a test of their own.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module undetermined.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass loud(T) <= show(T) where [ func loud(T) = string ].
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- instance show(int) where [ show(X) = string.int_to_string(X) ].
:- instance show(list(T)) <= show(T) where [ show(_) = "[...]" ].
:- instance conv(int, string) where [ conv(X) = string.int_to_string(X) ].

main(0).
% mmc reports these at the declaration, and mostly the goal as their
% cause.
:- func undetermined = string. % mmc-differs
undetermined = show(_).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:16: error: unsatisfied type class constraint `undetermined.show(T_2)': nothing determines its type
:- func undetermined_list = string. % mmc-differs
undetermined_list = show([]). % mmc-differs
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:21: error: unsatisfied type class constraint `undetermined.show(T_3)': nothing determines its type
% CHECK: [[@LINE-3]]:21: because: an instance for `undetermined.show(list.list(T_3))' needs it
:- func undetermined_multi = int. % mmc-differs
undetermined_multi = 0 :- _ = conv(1).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:31: error: unsatisfied type class constraint `undetermined.conv(int, T_4)': nothing determines its type
