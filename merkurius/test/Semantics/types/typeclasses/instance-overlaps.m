% Instances that overlap (reference manual, "Instance declarations"): two
% for the same types, up to renaming, also through an equivalence type,
% or abstract and with methods in the interface.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module instance_overlaps.
:- interface.
:- import_module list.
:- typeclass d(T) where [].
:- typeclass e(T, U) where [].
:- instance d(int) where [].
% CHECK: [[@LINE-1]]:13: error: the instance instance_overlaps.d(int) is not abstract, so it may not be in the interface section
:- instance d(list(T)).
:- implementation.

:- type p(A, B) ---> p(A, B).
:- type e == int.
:- instance d(e) where [].
% CHECK: [[@LINE-1]]:13: error: instance instance_overlaps.d(int) overlaps an earlier instance of typeclass d/1
% CHECK: [[@LINE-9]]:13: because: the earlier instance is instance_overlaps.d(int)
:- instance d(list(U)) where [].
% Both apply to p(int, int). mmc misses the overlap, and the C compiler
% then rejects its code (see the design document's register).
:- instance d(p(A, B)) where []. % mmc-differs
:- instance d(p(A, A)) where []. % mmc-differs
% CHECK: [[@LINE-1]]:13: error: instance instance_overlaps.d(instance_overlaps.p(A, A)) overlaps an earlier instance of typeclass d/1
% CHECK: [[@LINE-3]]:13: because: the earlier instance is instance_overlaps.d(instance_overlaps.p(A, B))
:- instance e(int, p(A, B)) where [].
:- instance e(int, p(B, A)) where [].
% CHECK: [[@LINE-1]]:13: error: instance instance_overlaps.e(int, instance_overlaps.p(B, A)) overlaps an earlier instance of typeclass e/2
% CHECK: [[@LINE-3]]:13: because: the earlier instance is instance_overlaps.e(int, instance_overlaps.p(A, B))
:- instance e(int, list(A)) where [].
:- instance e(list(A), int) where [].
