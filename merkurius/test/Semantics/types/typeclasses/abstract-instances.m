% Abstract instances (reference manual, "Abstract instance
% declarations"): each needs a definition in the implementation section,
% with the same constraints, up to renaming.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module abstract_instances.
:- interface.
:- import_module list.
:- typeclass d(T) where [].
:- typeclass e(T, U) where [].
:- type maybe(T) ---> no ; yes(T).
:- type t ---> t.
:- instance d(int) where [].
% CHECK: [[@LINE-1]]:13: error: the instance abstract_instances.d(int) is not abstract, so it may not be in the interface section
:- instance d(string).
:- instance d(list(T)) <= d(T).
:- instance d(maybe(T)) <= d(T).
:- instance d(t).
% CHECK: [[@LINE-1]]:13: error: the abstract instance abstract_instances.d(abstract_instances.t) has no definition in the implementation section
:- instance e(list(T), list(U)) <= (d(T), d(U)).
:- implementation.

:- instance d(string) where [].
:- instance d(list(U)) <= d(U) where [].
:- instance d(maybe(T)) where [].
% CHECK: [[@LINE-1]]:13: error: the constraints of instance abstract_instances.d(abstract_instances.maybe(T)) differ from those of its abstract declaration
% CHECK: [[@LINE-10]]:13: because: the abstract declaration is here
:- instance e(list(A), list(B)) <= d(A) where [].
% CHECK: [[@LINE-1]]:13: error: the constraints of instance abstract_instances.e(list.list(A), list.list(B)) differ from those of its abstract declaration
% CHECK: [[@LINE-10]]:13: because: the abstract declaration is here
