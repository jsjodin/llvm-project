% Type-correct coerces (reference manual, "Type conversions"): both types
% are discriminated union types with the same base type, whose arguments
% are the same where the base type's definition needs them to be, and
% otherwise one a subtype of the other. Coercing to a subtype is
% type-correct, though mode analysis may reject it (f2, down and sibling).
% coerce determines neither type, but an assignment whose coerce is
% invalid is dropped, as in mmc (chosen and chosen_result).
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module coerce.
:- interface.
:- import_module list, maybe, pair.
:- pred main(int::out) is det.
:- implementation.
:- import_module int.

main(0).

% The manual's examples.
:- type fruit
    --->    apple
    ;       lemon
    ;       orange.
:- type citrus =< fruit
    --->    lemon
    ;       orange.
:- func f1(citrus) = fruit.
f1(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.f1/1: func coerce.f1(coerce.citrus) = coerce.fruit
:- func f2(fruit) = citrus.
f2(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.f2/1: func coerce.f2(coerce.fruit) = coerce.citrus
:- inst citrus for fruit/0
    --->    lemon
    ;       orange.
:- func f3(fruit) = citrus.
:- mode f3(in(citrus)) = out is det.
f3(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.f3/1: func coerce.f3(coerce.fruit) = coerce.citrus

% A parameter that occurs only as an argument, or in the type's own
% recursion, may be a subtype in one and its supertype in the other.
:- type wrap(T) ---> wrap(T).
:- func up(wrap(citrus)) = wrap(fruit).
up(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.up/1: func coerce.up(coerce.wrap(coerce.citrus)) = coerce.wrap(coerce.fruit)
:- func down(wrap(fruit)) = wrap(citrus).
down(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.down/1: func coerce.down(coerce.wrap(coerce.fruit)) = coerce.wrap(coerce.citrus)
:- func same(fruit) = fruit.
same(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.same/1: func coerce.same(coerce.fruit) = coerce.fruit
:- func of_list(list(citrus)) = list(fruit).
of_list(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.of_list/1: func coerce.of_list(list.list(coerce.citrus)) = list.list(coerce.fruit)
:- func of_maybe(maybe(citrus)) = maybe(fruit).
of_maybe(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.of_maybe/1: func coerce.of_maybe(maybe.maybe(coerce.citrus)) = maybe.maybe(coerce.fruit)
:- func of_pair(pair(citrus, int)) = pair(fruit, int).
of_pair(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.of_pair/1: func coerce.of_pair(pair.pair(coerce.citrus, int)) = pair.pair(coerce.fruit, int)
:- type phantom(T) ---> phantom.
:- func of_phantom(phantom(citrus)) = phantom(fruit).
of_phantom(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.of_phantom/1: func coerce.of_phantom(coerce.phantom(coerce.citrus)) = coerce.phantom(coerce.fruit)
:- type ex(T) ---> ex(T) ; some [U] ex2(U, T).
:- func of_ex(ex(citrus)) = ex(fruit).
of_ex(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.of_ex/1: func coerce.of_ex(coerce.ex(coerce.citrus)) = coerce.ex(coerce.fruit)

% Between subtypes: the base types are compared, with their arguments.
:- type pome =< fruit ---> apple.
:- func sibling(citrus) = pome.
sibling(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.sibling/1: func coerce.sibling(coerce.citrus) = coerce.pome
:- type non_empty(T) =< list(T) ---> [T | list(T)].
:- func to_list(non_empty(T)) = list(T).
to_list(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.to_list/1: func coerce.to_list(coerce.non_empty(T)) = list.list(T)
:- func to_fruits(non_empty(citrus)) = list(fruit).
to_fruits(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.to_fruits/1: func coerce.to_fruits(coerce.non_empty(coerce.citrus)) = list.list(coerce.fruit)
:- type listed(T) =< wrap(list(T)) ---> wrap(list(T)).
:- func unlisted(listed(citrus)) = wrap(list(fruit)).
unlisted(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.unlisted/1: func coerce.unlisted(coerce.listed(coerce.citrus)) = coerce.wrap(list.list(coerce.fruit))
:- type wrapped(T) == wrap(T).
:- func unwrapped(wrapped(citrus)) = wrap(fruit).
unwrapped(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.unwrapped/1: func coerce.unwrapped(coerce.wrap(coerce.citrus)) = coerce.wrap(coerce.fruit)
:- func polymorphic(wrap(T)) = wrap(T).
polymorphic(X) = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of func coerce.polymorphic/1: func coerce.polymorphic(coerce.wrap(T)) = coerce.wrap(T)

% Overloading: only aa's k can be coerced to aa, or from aa.
:- type aa ---> k ; j.
:- type bb ---> k.
:- pred chosen(aa::out) is det.
chosen(Y) :- X = k, Y = coerce(X).
% CHECK: clause at [[@LINE-1]]:1 of pred coerce.chosen/1: pred coerce.chosen(coerce.aa)
% CHECK-NEXT: Y: coerce.aa
% CHECK-NEXT: X: coerce.aa
:- pred chosen_result(int::out) is det.
chosen_result(0) :- X = (k : aa), Y = coerce(X), Y = k.
% CHECK: clause at [[@LINE-1]]:1 of pred coerce.chosen_result/1: pred coerce.chosen_result(int)
% CHECK-NEXT: X: coerce.aa
% CHECK-NEXT: Y: coerce.aa

% The context determines the types.
:- func in_list(citrus) = list(fruit).
in_list(X) = [coerce(X)].
% CHECK: clause at [[@LINE-1]]:1 of func coerce.in_list/1: func coerce.in_list(coerce.citrus) = list.list(coerce.fruit)
:- func in_lambda(citrus) = fruit.
in_lambda(X) = Y :-
% CHECK: clause at [[@LINE-1]]:1 of func coerce.in_lambda/1: func coerce.in_lambda(coerce.citrus) = coerce.fruit
    F = (func(A) = coerce(A)),
    Y = apply(F, X).
