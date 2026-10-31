% The rules of typeclass declarations (reference manual, "Typeclass
% declarations", "Abstract typeclass declarations" and "Type class
% constraints on type class declarations").
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module class_errors.
:- interface.
:- typeclass base(T) where [].
:- typeclass twice(T) where [ func twice(T) = int ].
:- typeclass abstract_only(T).
% CHECK: [[@LINE-1]]:14: error: the abstract typeclass abstract_only/1 has no definition
:- typeclass differs(T).
:- typeclass same(T) <= base(T).
:- implementation.
:- import_module list.

:- typeclass repeated(T, T) where [].
% CHECK: [[@LINE-1]]:14: error: the parameter T appears more than once
:- typeclass free(T) <= base(U) where [].
% CHECK: [[@LINE-1]]:25: error: type variable U in the superclass constraint is not a parameter of the type class
:- typeclass nonground(T) <= base(list(T)) where [].
% CHECK: [[@LINE-1]]:30: error: a superclass constraint may only constrain type variables and ground types
% The manual requires a variable in each constraint, mmc does not (see the
% design document's register).
:- typeclass ground(T) <= base(int) where [].
:- typeclass a(T) <= b(T) where [].
% CHECK: [[@LINE-1]]:14: error: the superclass relation is cyclic: class_errors.a/1 <= class_errors.b/1 <= class_errors.c/1 <= class_errors.a/1
:- typeclass b(T) <= c(T) where [].
:- typeclass c(T) <= a(T) where [].
:- typeclass self(T) <= self(T) where [].
% CHECK: [[@LINE-1]]:14: error: the superclass relation is cyclic: class_errors.self/1 <= class_errors.self/1
:- typeclass undetermined(T) where [ func f(int) = int, func g(T) = int ].
% CHECK: [[@LINE-1]]:38: error: type variable T occurs in the constraints of function f/1, but not in its argument or result types
:- typeclass twice(T) where [ func twice(T) = int ].
% CHECK: [[@LINE-1]]:14: error: typeclass twice/1 is defined more than once
% CHECK: [[@LINE-27]]:14: because: it is first defined here
:- typeclass differs(T) <= base(T) where [].
% CHECK: [[@LINE-1]]:14: error: the superclass constraints of typeclass differs/1 differ from those of its abstract declaration
% CHECK: [[@LINE-27]]:14: because: the abstract declaration is here
% Parameters may be renamed.
:- typeclass same(U) <= base(U) where [].
:- pred p(int::in) is det <= base(T).
% CHECK: [[@LINE-1]]:4: error: type variable T occurs in the constraints of predicate p/1, but not in its argument types
p(_).
:- func q(list(T)) = int <= base(T).
q(_) = 0.
