% A type whose only value is a constant cannot have a user-defined equality
% or comparison (reference manual, "User-defined equality and comparison").
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module user_equality_unit.
:- interface.
:- type unused ---> unused.
:- implementation.
:- type unit ---> unit where equality is unit_equals.
% CHECK: [[@LINE-1]]:4: error: the type unit/0 has only one value, so it cannot have a user-defined equality or comparison
:- pred unit_equals(unit::in, unit::in) is semidet.
unit_equals(_, _) :- semidet_true.

:- type unit2 ---> unit2 where comparison is unit2_compare.
% CHECK: [[@LINE-1]]:4: error: the type unit2/0 has only one value, so it cannot have a user-defined equality or comparison
:- pred unit2_compare(comparison_result::uo, unit2::in, unit2::in) is det.
unit2_compare((=), _, _).
