% The types of an instance (reference manual, "Instance declarations"):
% each a type constructor applied to type variables, after equivalence
% types are expanded.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module instance_types.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list.

:- typeclass d(T) where [].
:- typeclass e(T, U) where [].
:- type p(A, B) ---> p(A, B).
:- type q(A) ---> q(A).
:- type r(A) ---> r(A).
:- type s(A) ---> s(A).
:- type rint == r(int).
:- type sint == s(int).
:- type t ---> t.
:- type tt == t.

main(0).
:- instance d(T) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d(T), type 1, `T', is a type variable; it must be a type constructor applied to type variables
:- instance d(list(list(T))) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d(list.list(list.list(T))), argument 1 of type 1, `list.list(T)', is not a type variable
:- instance d(rint) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d(instance_types.r(int)), argument 1 of type 1, `int', is not a type variable
:- instance d({int, A}) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d({int, A}), argument 1 of type 1, `int', is not a type variable
:- instance d(pred(A)) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d(pred(A)), type 1, `pred(A)', is a higher-order type; it must be a type constructor applied to type variables
:- instance d((func(A) = B)) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.d((func(A) = B)), type 1, `(func(A) = B)', is a higher-order type; it must be a type constructor applied to type variables
:- instance e(int, q(int)) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.e(int, instance_types.q(int)), argument 1 of type 2, `int', is not a type variable
:- instance e(sint, U) where [].
% CHECK: [[@LINE-1]]:13: error: in instance instance_types.e(instance_types.s(int), U), argument 1 of type 1, `int', is not a type variable
% CHECK: [[@LINE-2]]:13: error: in instance instance_types.e(instance_types.s(int), U), type 2, `U', is a type variable; it must be a type constructor applied to type variables
% Allowed: a repeated variable, a tuple, an equivalence type and several
% types.
:- instance d(p(T, T)) where [].
:- instance d({A, B, C}) where [].
:- instance d(tt) where [].
:- instance e(list(A), p(B, A)) where [].
