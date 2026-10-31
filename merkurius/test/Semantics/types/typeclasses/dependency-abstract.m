% An abstract typeclass declaration has the functional dependencies of its
% definition (reference manual, "Functional dependencies"); the order of
% the dependencies and of their variables does not matter.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module dependency_abstract.
:- interface.
:- typeclass more(A, B) <= (A -> B).
:- typeclass fewer(A, B).
:- typeclass other(A, B) <= (A -> B).
:- typeclass same(A, B, C) <= ((A, B -> C), (C -> A)).
:- implementation.
:- typeclass more(A, B) where [].
% CHECK: [[@LINE-1]]:14: error: the functional dependencies of typeclass more/2 differ from those of its abstract declaration
% CHECK: [[@LINE-7]]:14: because: the abstract declaration is here
:- typeclass fewer(A, B) <= (A -> B) where [].
% CHECK: [[@LINE-1]]:14: error: the functional dependencies of typeclass fewer/2 differ from those of its abstract declaration
% CHECK: [[@LINE-9]]:14: because: the abstract declaration is here
:- typeclass other(A, B) <= (B -> A) where [].
% CHECK: [[@LINE-1]]:14: error: the functional dependencies of typeclass other/2 differ from those of its abstract declaration
% CHECK: [[@LINE-11]]:14: because: the abstract declaration is here
:- typeclass same(A, B, C) <= ((C -> A), (B, A -> C)) where [].
