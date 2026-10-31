% The rules for existentially typed constructors (reference manual,
% "Existentially typed data types"): no type variable is both a parameter
% of the type and existentially quantified; each existentially quantified
% variable occurs in the arguments or in the constraints, and those of the
% constraints are existentially quantified and determined by the
% arguments' types, perhaps through functional dependencies.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module type_definition_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, string.

main(0).
:- typeclass showable(T) where [ func show(T) = string ].
:- typeclass xable(A, B) where [].
:- typeclass yable(A, B) <= (A -> B) where [].

:- type t12(T) ---> f1(T) ; some [T] f2(T).
% CHECK: [[@LINE-1]]:29: error: type variable T of constructor f2/1 is a parameter of its type, and cannot be existentially quantified too
:- type t3 ---> some [T1, T2] f3(T1) => xable(T1, T2).
% CHECK: [[@LINE-1]]:17: error: type variable T2 occurs in the constraints of constructor f3/1, but its argument types do not determine it
:- type t4(U) ---> some [T] f4(T, U) => showable(U).
% CHECK: [[@LINE-1]]:20: error: type variable U is in a constraint of constructor f4/2, but is not existentially quantified
:- type t5(U) ---> f5(U) => showable(U).
% CHECK: [[@LINE-1]]:20: error: type variable U is in a constraint of constructor f5/1, but is not existentially quantified
:- type t6 ---> some [T] f6(int).
% CHECK: [[@LINE-1]]:17: error: existentially quantified type variable T of constructor f6/1 occurs neither in its arguments nor in its constraints
% A functional dependency determines T2 from T1.
:- type t7 ---> some [T1, T2] f7(T1) => yable(T1, T2).
