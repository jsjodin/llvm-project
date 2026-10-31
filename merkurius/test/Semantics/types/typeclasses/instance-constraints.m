% The constraints of an instance (reference manual, "Type class
% constraints on instance declarations"): on variables of its types or on
% ground types.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module instance_constraints.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list.

:- typeclass d(T) where [].
:- type q(A) ---> q(A).
:- type r(A) ---> r(A).
:- type s(A) ---> s(A).
:- type t(A, B) ---> t(A, B).

main(0).
:- instance d(q(T)) <= d(U) where [].
% CHECK: [[@LINE-1]]:24: error: type variable U in the constraints of instance instance_constraints.d(instance_constraints.q(T)) does not occur in its types
:- instance d(r(T)) <= d(list(T)) where [].
% CHECK: [[@LINE-1]]:24: error: an instance's constraint may only constrain type variables and ground types
% The manual requires a variable in each constraint, mmc does not (see the
% design document's register).
:- instance d(s(T)) <= d(int) where [].
:- instance d(t(A, B)) <= (d(A), d(B)) where [].
:- instance d(list(T)) <= d(T) where [].
