% A constraint that inference cannot put in a signature, since it is on a
% type variable that the head does not have: nothing determines it.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>/dev/null | FileCheck --check-prefix=TYPES %s
% RUN: %if mmc %{ %compare-types --allow-errors %s %}

:- module inference_undetermined.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- instance show(int) where [ show(X) = string.int_to_string(X) ].
:- instance show(list(T)) <= show(T) where [ show(_) = "[...]" ].

main(0).
local(X) = 0 :- _ = conv(X).
% CHECK: [[@LINE-1]]:1: warning: unresolved polymorphism: the type variables of these types are not determined, and will be bound to void:
% CHECK: [[@LINE-2]]:21: error: unsatisfied type class constraint `inference_undetermined.conv(T1, T_5)': nothing determines its type
% TYPES: clause at {{[0-9]+}}:1 of func inference_undetermined.local/1: inferred func inference_undetermined.local(T1) = int
