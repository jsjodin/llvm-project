% The predicates a type names for its equality and comparison (reference
% manual, "User-defined equality and comparison"): each must be of the
% type's module and take values of the type, perhaps at a more general
% type, and a comparison_result first for a comparison. mmc reports each at
% the type. user-equality-unit.m and user-equality-purity.m have the errors
% of mmc's other passes.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module user_equality.
:- interface.
:- type unused ---> unused.
:- implementation.
:- import_module list.

% Accepted: polymorphic and more general predicates.
:- type set(T) ---> set(list(T))
    where equality is set_equals, comparison is set_compare.
:- pred set_equals(set(T)::in, set(T)::in) is semidet.
set_equals(_, _) :- semidet_true.
:- pred set_compare(comparison_result::uo, set(T)::in, set(T)::in) is det.
set_compare((=), _, _).
:- type token ---> token(int) where equality is any_equal.
:- pred any_equal(T::in, T::in) is semidet.
any_equal(_, _) :- semidet_true.

:- type box ---> box(int) where equality is missing_equals.
% CHECK: [[@LINE-1]]:4: error: the equality predicate user_equality.missing_equals/2 of type box/0 is not defined in its module

:- type box2 ---> box2(int) where equality is int_equals.
% CHECK: [[@LINE-1]]:4: error: the equality predicate int_equals/2 of type box2/0 does not take two values of the type
:- pred int_equals(int::in, int::in) is semidet.
int_equals(X, X).

:- type box3 ---> box3(int) where comparison is box3_compare.
% CHECK: [[@LINE-1]]:4: error: the comparison predicate box3_compare/3 of type box3/0 does not take a comparison_result and two values of the type
:- pred box3_compare(box3::in, box3::in, comparison_result::uo) is det.
box3_compare(_, _, (=)).
