% The rules for subtypes (reference manual, "Subtypes"): the supertype is a
% discriminated union type, and the chain of supertypes ends at a base
% type; the subtype's constructors are among its supertype's, at most once
% each, with arguments whose types are subtypes of the supertype's, and
% the same existential variables and constraints; and a subtype has no
% equality or comparison of its own. Constructors in a different order
% from the supertype's draw a warning, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --error-lines %s %}

:- module subtype_errors.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module int.

main(0).
:- type fruit ---> apple ; pear ; lemon ; orange.
:- type basket ---> basket(fruit, int).
:- type citrus =< fruit ---> lemon ; orange.

:- type not_a_constructor =< fruit ---> lemon ; banana.
% CHECK: [[@LINE-1]]:49: error: banana/0 is not a constructor of the supertype subtype_errors.fruit
:- type other_arity =< fruit ---> lemon(int).
% CHECK: [[@LINE-1]]:35: error: lemon/1 is not a constructor of the supertype subtype_errors.fruit
:- type wider_argument =< basket ---> basket(int, int).
% CHECK: [[@LINE-1]]:39: error: argument 1 of basket/2 has type int, which is not a subtype of subtype_errors.fruit, its type in the supertype
:- type not_in_citrus =< citrus ---> lemon ; apple.
% CHECK: [[@LINE-1]]:46: error: apple/0 is not a constructor of the supertype subtype_errors.citrus
:- type of_int =< int ---> foo.
% CHECK: [[@LINE-1]]:4: error: the supertype int of of_int/0 is not a discriminated union type whose definition is visible
:- type circular_a =< circular_b ---> x.
% CHECK: [[@LINE-1]]:4: error: the chain of supertypes of circular_a/0 is circular
% CHECK: [[@LINE+1]]:9: because: it goes through subtype_errors.circular_b/0
:- type circular_b =< circular_a ---> x.
% CHECK: [[@LINE-1]]:4: error: the chain of supertypes of circular_b/0 is circular
% CHECK: [[@LINE-5]]:9: because: it goes through subtype_errors.circular_a/0
:- type circular_self =< circular_self ---> x.
% CHECK: [[@LINE-1]]:4: error: the chain of supertypes of circular_self/0 is circular
:- type reordered =< fruit ---> orange ; lemon.
% CHECK: [[@LINE-1]]:4: warning: the constructors of reordered/0 are in a different order from those of its supertype subtype_errors.fruit
:- type twice =< fruit ---> lemon ; orange ; lemon.
% CHECK: [[@LINE-1]]:4: warning: the constructors of twice/0 are in a different order from those of its supertype subtype_errors.fruit
% CHECK: [[@LINE-2]]:46: error: constructor lemon/0 is defined twice in type twice/0
:- type with_equality =< fruit ---> lemon where equality is eq_lemon.
% CHECK: [[@LINE-1]]:4: error: the subtype with_equality/0 cannot have its own equality or comparison; it has its base type's
:- pred eq_lemon(with_equality::in, with_equality::in) is semidet.
eq_lemon(lemon, lemon).

:- typeclass fooable(T) where [].
:- type maybe_foo ---> none ; some [T] foo(T) => fooable(T).
:- type unconstrained =< maybe_foo ---> some [T] foo(T).
% CHECK: [[@LINE-1]]:41: error: foo/1 must have the existentially quantified type variables and constraints of the supertype's constructor, one to one
:- type not_existential =< maybe_foo ---> foo(int).
% CHECK: [[@LINE-1]]:43: error: argument 1 of foo/1 has type int, which is not a subtype of T, its type in the supertype
% CHECK: [[@LINE-2]]:43: error: foo/1 must have the existentially quantified type variables and constraints of the supertype's constructor, one to one
