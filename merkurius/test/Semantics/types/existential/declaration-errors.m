% An existential constraint constrains only existentially quantified type
% variables, and a universal one only universally quantified ones
% (reference manual, "Existential class constraints"). A type variable
% quantified twice is an error too; mmc crashes on it (see the register).
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module declaration_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, string.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- instance showable(int) where [ show(X) = int_to_string(X) ].

:- some [T] pred universal_constrained(T::out, U::in) is det => showable(U).
% CHECK: [[@LINE-1]]:4: error: type variable U of predicate universal_constrained/2 is existentially constrained, but universally quantified
universal_constrained(1, _).
:- some [T] pred existential_constrained(T::out) is det <= showable(T).
% CHECK: [[@LINE-1]]:4: error: type variable T of predicate existential_constrained/1 is universally constrained, but existentially quantified
existential_constrained(1).
:- pred unquantified(T::out) is det => showable(T).
% CHECK: [[@LINE-1]]:4: error: type variable T of predicate unquantified/1 is existentially constrained, but universally quantified
unquantified(_).
% Quantified variables need not occur in the arguments.
:- some [T] pred unused(int::out) is det.
unused(1).
