% Subtypes (reference manual, "Subtypes"): a subtype's constructors are
% some of its supertype's, each argument's type a subtype of the
% supertype's, by a chain of definitions that ends at a base type. The
% examples are the manual's. A subtype is a type of its own, whose
% constructors overload its supertype's, and its values are not its
% supertype's without a coerce.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types %s %}

:- module subtypes.
:- interface.
:- import_module list.
:- pred main(int::out) is det.
% A subtype may be exported abstractly.
:- type non_empty_list(T).
:- implementation.
:- import_module int.

main(0).
:- type non_empty_list(T) =< list(T)
    --->    [T | list(T)].

:- type fruit
    --->    apple
    ;       pear
    ;       lemon
    ;       orange.
:- type citrus_fruit =< fruit
    --->    lemon
    ;       orange.
:- type fruit_basket
    --->    basket(fruit, int).
:- type citrus_fruit_basket =< fruit_basket
    --->    basket(citrus_fruit, int).
% A subtype of a subtype, and one with a phantom parameter.
:- type lemon_only =< citrus_fruit ---> lemon.
:- type tagged(T) =< fruit ---> apple.

:- type foo ---> foo.
:- type non_empty_list_of_foo =< list(foo)
    --->    [foo | list(foo)].
:- typeclass fooable(T) where [].
:- type maybe_foo
    --->    none
    ;       some [T] foo(T) => fooable(T).
:- type some_foo =< maybe_foo
    --->    some [T] foo(T) => fooable(T).
:- type task
    --->    create(pred(int))
    ;       delete(pred(int)).
:- type create_task =< task
    --->    create(pred(int)).

:- type source_id == int.
:- type or_node
    --->    or_source(source_id)
    ;       logical_or(and_node, and_node)
    ;       logical_or_list(and_node, and_node, and_node, list(and_node)).
:- type and_node
    --->    and_source(source_id)
    ;       logical_and(or_node, or_node)
    ;       logical_and_list(or_node, or_node, or_node, list(or_node)).
:- type binary_or_node =< or_node
    --->    or_source(source_id)
    ;       logical_or(binary_or_and_node, binary_or_and_node).
:- type binary_or_and_node =< and_node
    --->    and_source(source_id)
    ;       logical_and(binary_or_node, binary_or_node)
    ;       logical_and_list(binary_or_node, binary_or_node, binary_or_node,
                list(binary_or_node)).

% The declared types choose among the overloaded constructors.
:- func a_lemon = citrus_fruit.
a_lemon = lemon.
% CHECK: clause at [[@LINE-1]]:1 of func subtypes.a_lemon/0: func subtypes.a_lemon = subtypes.citrus_fruit
% CHECK: [[@LINE-2]]:11 lemon -> constructor subtypes.lemon/0
:- func a_basket = citrus_fruit_basket.
a_basket = basket(orange, 3).
% CHECK: clause at [[@LINE-1]]:1 of func subtypes.a_basket/0: func subtypes.a_basket = subtypes.citrus_fruit_basket
:- func first(non_empty_list(T)) = T.
first([X | _]) = X.
% CHECK: clause at [[@LINE-1]]:1 of func subtypes.first/1: func subtypes.first(subtypes.non_empty_list(T)) = T
:- func count(citrus_fruit_basket) = int.
count(basket(_, N)) = N.
% CHECK: clause at [[@LINE-1]]:1 of func subtypes.count/1: func subtypes.count(subtypes.citrus_fruit_basket) = int
% Inference: the constructor's argument decides which basket it is.
inferred_count(basket(F, N)) = N :- F = (F : citrus_fruit).
% CHECK: clause at [[@LINE-1]]:1 of func subtypes.inferred_count/1: inferred func subtypes.inferred_count(subtypes.citrus_fruit_basket) = int
