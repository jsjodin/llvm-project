% Coerces that are not type-correct (reference manual, "Type
% conversions"): types that are not discriminated union types, different
% base types, arguments neither of which is a subtype of the other, and
% different arguments for a parameter that the base type's definition
% uses in a higher-order type or another type. Nor does coerce determine
% a type that nothing else does.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module coerce_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int, list.

main(0).
:- type fruit ---> apple ; lemon ; orange.
:- type citrus =< fruit ---> lemon ; orange.
:- type veg ---> carrot.
:- type wrap(T) ---> wrap(T).

% The manual's example: T of wrap/1 is fruit in one and citrus in the
% other, which are not types that coerce can relate inside func types.
:- func f4(func(fruit) = int) = (func(citrus) = int).
f4(X) = Y :- wrap(Y) = coerce(wrap(X)).
% CHECK: [[@LINE-1]]:24: error: cannot coerce from coerce_errors.wrap((func(coerce_errors.fruit) = int)) to coerce_errors.wrap((func(coerce_errors.citrus) = int))
% CHECK: [[@LINE-2]]:24: because: the parameter T of coerce_errors.wrap/1, (func(coerce_errors.fruit) = int) in one and (func(coerce_errors.citrus) = int) in the other, and neither is a subtype of the other

% Not discriminated union types.
:- func of_int(int) = int.
of_int(X) = coerce(X).
% CHECK: [[@LINE-1]]:13: error: cannot coerce from int to int
% CHECK: [[@LINE-2]]:13: because: int is not a discriminated union type whose definition is visible
:- func to_int(citrus) = int.
to_int(X) = coerce(X).
% CHECK: [[@LINE-1]]:13: error: cannot coerce from coerce_errors.citrus to int
% CHECK: [[@LINE-2]]:13: because: int is not a discriminated union type whose definition is visible
:- func of_tuple({citrus}) = {fruit}.
of_tuple(X) = coerce(X).
% CHECK: [[@LINE-1]]:15: error: cannot coerce from {coerce_errors.citrus} to {coerce_errors.fruit}
% CHECK: [[@LINE-2]]:15: because: {coerce_errors.citrus} is not a discriminated union type whose definition is visible
:- func of_func(func(citrus) = int) = (func(citrus) = int).
of_func(X) = coerce(X).
% CHECK: [[@LINE-1]]:14: error: cannot coerce from (func(coerce_errors.citrus) = int) to (func(coerce_errors.citrus) = int)
% CHECK: [[@LINE-2]]:14: because: (func(coerce_errors.citrus) = int) is not a discriminated union type whose definition is visible
:- func of_variable(T) = T.
of_variable(X) = coerce(X).
% CHECK: [[@LINE-1]]:18: error: cannot coerce from T to T
% CHECK: [[@LINE-2]]:18: because: T is not a discriminated union type whose definition is visible

% Different base types, and unrelated arguments.
:- func unrelated(fruit) = veg.
unrelated(X) = coerce(X).
% CHECK: [[@LINE-1]]:16: error: cannot coerce from coerce_errors.fruit to coerce_errors.veg
% CHECK: [[@LINE-2]]:16: because: their base types, coerce_errors.fruit and coerce_errors.veg, differ
:- func unrelated_arguments(wrap(veg)) = wrap(fruit).
unrelated_arguments(X) = coerce(X).
% CHECK: [[@LINE-1]]:26: error: cannot coerce from coerce_errors.wrap(coerce_errors.veg) to coerce_errors.wrap(coerce_errors.fruit)
% CHECK: [[@LINE-2]]:26: because: the parameter T of coerce_errors.wrap/1, coerce_errors.veg in one and coerce_errors.fruit in the other, and neither is a subtype of the other

% Parameters that must be the same: in another discriminated union type,
% also through an equivalence, in a higher-order type, in a recursion
% that is not of the exact form, and in mutually recursive types.
:- type listed(T) ---> listed(list(T)).
:- func of_listed(listed(citrus)) = listed(fruit).
of_listed(X) = coerce(X).
% CHECK: [[@LINE-1]]:16: error: cannot coerce from coerce_errors.listed(coerce_errors.citrus) to coerce_errors.listed(coerce_errors.fruit)
% CHECK: [[@LINE-2]]:16: because: the parameter T of coerce_errors.listed/1, coerce_errors.citrus in one and coerce_errors.fruit in the other, must be the same type in both, since it occurs in list.list(T) in its definition
:- type list_of(T) == list(T).
:- type equivalent(T) ---> equivalent(list_of(T)).
:- func of_equivalent(equivalent(citrus)) = equivalent(fruit).
of_equivalent(X) = coerce(X).
% CHECK: [[@LINE-1]]:20: error: cannot coerce from coerce_errors.equivalent(coerce_errors.citrus) to coerce_errors.equivalent(coerce_errors.fruit)
% CHECK: [[@LINE-2]]:20: because: the parameter T of coerce_errors.equivalent/1, coerce_errors.citrus in one and coerce_errors.fruit in the other, must be the same type in both, since it occurs in list.list(T) in its definition
:- type function(T) ---> function(func(T) = int).
:- func of_function(function(citrus)) = function(fruit).
of_function(X) = coerce(X).
% CHECK: [[@LINE-1]]:18: error: cannot coerce from coerce_errors.function(coerce_errors.citrus) to coerce_errors.function(coerce_errors.fruit)
% CHECK: [[@LINE-2]]:18: because: the parameter T of coerce_errors.function/1, coerce_errors.citrus in one and coerce_errors.fruit in the other, must be the same type in both, since it occurs in (func(T) = int) in its definition
:- type swapped(A, B) ---> swapped(A, swapped(B, A)) ; none.
:- func of_swapped(swapped(citrus, int)) = swapped(fruit, int).
of_swapped(X) = coerce(X).
% CHECK: [[@LINE-1]]:17: error: cannot coerce from coerce_errors.swapped(coerce_errors.citrus, int) to coerce_errors.swapped(coerce_errors.fruit, int)
% CHECK: [[@LINE-2]]:17: because: the parameter A of coerce_errors.swapped/2, coerce_errors.citrus in one and coerce_errors.fruit in the other, must be the same type in both, since it occurs in coerce_errors.swapped(B, A) in its definition
:- type ta(T) ---> ta(T) ; tb(tb(T)).
:- type tb(T) ---> tbb(ta(T)).
:- func of_ta(ta(citrus)) = ta(fruit).
of_ta(X) = coerce(X).
% CHECK: [[@LINE-1]]:12: error: cannot coerce from coerce_errors.ta(coerce_errors.citrus) to coerce_errors.ta(coerce_errors.fruit)
% CHECK: [[@LINE-2]]:12: because: the parameter T of coerce_errors.ta/1, coerce_errors.citrus in one and coerce_errors.fruit in the other, must be the same type in both, since it occurs in coerce_errors.tb(T) in its definition

% Types that nothing determines; coerce determines neither.
:- pred unknown_result(citrus::in) is det.
unknown_result(X) :- Y = coerce(X), _ = Y.
% CHECK: [[@LINE-1]]:26: error: cannot coerce from coerce_errors.citrus to T_1
% CHECK: [[@LINE-2]]:26: because: nothing determines the type of its result, and coerce does not
:- func unknown_argument = fruit.
unknown_argument = coerce(_).
% CHECK: [[@LINE-1]]:20: error: cannot coerce from T_1 to coerce_errors.fruit
% CHECK: [[@LINE-2]]:20: because: nothing determines the type of its argument, and coerce does not
:- pred twice(fruit::out) is det.
twice(Z) :- X = (lemon : citrus), Y = coerce(X), Z = coerce(Y).
% CHECK: [[@LINE-1]]:39: error: cannot coerce from coerce_errors.citrus to T_2
% CHECK: [[@LINE-2]]:39: because: nothing determines the type of its result, and coerce does not
% CHECK: [[@LINE-3]]:54: error: cannot coerce from T_2 to coerce_errors.fruit
% CHECK: [[@LINE-4]]:54: because: nothing determines the type of its argument, and coerce does not

% Both lemons can be coerced to a fruit.
:- pred ambiguous(fruit::out) is det.
ambiguous(Y) :- X = lemon, Y = coerce(X).
% CHECK: [[@LINE-1]]:1: error: ambiguous overloading causes type ambiguity
% CHECK: [[@LINE-2]]:21: because: lemon may be constructor coerce_errors.lemon/0 of type coerce_errors.fruit/0 or constructor coerce_errors.lemon/0 of type coerce_errors.citrus/0
