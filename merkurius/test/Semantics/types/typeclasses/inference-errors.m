% Constraints that inference cannot put in a signature: a ground one
% without an instance, also one that a callee's inferred constraint
% becomes. A constraint on a type that nothing determines is in
% inference-undetermined.m, since mmc reports those only in a module
% without these.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>/dev/null | FileCheck --check-prefix=TYPES %s
% RUN: %if mmc %{ %compare-types --allow-errors %s %}

:- module inference_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module float, int, list, string.

:- typeclass show(T) where [ func show(T) = string ].
:- typeclass conv(A, B) where [ func conv(A) = B ].
:- instance show(int) where [ show(X) = string.int_to_string(X) ].
:- instance show(list(T)) <= show(T) where [ show(_) = "[...]" ].

main(0).
no_instance(X) = show(X) ++ show([1.5]).
% CHECK: [[@LINE-1]]:29: error: unsatisfiable type class constraint `inference_errors.show(float)'
% CHECK: [[@LINE-2]]:29: because: an instance for `inference_errors.show(list.list(float))' needs it
simple(X) = show(X).
% TYPES: clause at [[@LINE-1]]:1 of func inference_errors.simple/1: inferred func inference_errors.simple(T1) = string <= (inference_errors.show(T1))
callee_float = simple(2.5).
% CHECK: [[@LINE-1]]:16: error: unsatisfiable type class constraint `inference_errors.show(float)'
% TYPES: clause at [[@LINE-2]]:1 of func inference_errors.callee_float/0: inferred func inference_errors.callee_float = string
