% A type's equality or comparison predicate must be pure (reference
% manual, "User-defined equality and comparison").
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module user_equality_purity.
:- interface.
:- type unused ---> unused.
:- implementation.
:- type box4 ---> box4(int) where equality is box4_equals.
% CHECK: [[@LINE-1]]:4: error: the equality predicate box4_equals/2 of type box4/0 must be pure
:- impure pred box4_equals(box4::in, box4::in) is semidet.
box4_equals(_, _) :- impure impure_true, semidet_true.
