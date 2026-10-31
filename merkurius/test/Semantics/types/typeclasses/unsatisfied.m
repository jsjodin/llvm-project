% Constraints of calls that are not satisfied: no instance, or a type
% variable that the declaration does not constrain, also after an
% instance reduces them. Each is reported once in a clause, at its first
% goal.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module unsatisfied.
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
:- func no_instance = string.
no_instance = show(1.5).
% CHECK: [[@LINE-1]]:15: error: unsatisfiable type class constraint `unsatisfied.show(float)'
:- func unconstrained(T) = string.
unconstrained(X) = show(X).
% CHECK: [[@LINE-1]]:20: error: unsatisfiable type class constraint `unsatisfied.show(T)'
:- func through_instance = string.
through_instance = show([1.5]).
% CHECK: [[@LINE-1]]:20: error: unsatisfiable type class constraint `unsatisfied.show(float)'
% CHECK: [[@LINE-2]]:20: because: an instance for `unsatisfied.show(list.list(float))' needs it
:- func through_instance_var(T) = string.
through_instance_var(X) = show([X]).
% CHECK: [[@LINE-1]]:27: error: unsatisfiable type class constraint `unsatisfied.show(T)'
% CHECK: [[@LINE-2]]:27: because: an instance for `unsatisfied.show(list.list(T))' needs it
% A subclass does not follow from its superclass.
:- func not_subclass(T) = string <= show(T).
not_subclass(X) = loud(X).
% CHECK: [[@LINE-1]]:19: error: unsatisfiable type class constraint `unsatisfied.loud(T)'
:- func once(float, float) = string.
once(X, Y) = show(X) ++ show(Y).
% CHECK: [[@LINE-1]]:14: error: unsatisfiable type class constraint `unsatisfied.show(float)'
:- func closure = (func(float) = string).
closure = show.
% CHECK: [[@LINE-1]]:11: error: unsatisfiable type class constraint `unsatisfied.show(float)'
:- func multi = float.
multi = conv(1).
% CHECK: [[@LINE-1]]:9: error: unsatisfiable type class constraint `unsatisfied.conv(int, float)'
% Undeclared.
inferred = show(1.5).
% CHECK: [[@LINE-1]]:12: error: unsatisfiable type class constraint `unsatisfied.show(float)'
