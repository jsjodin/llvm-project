% Mode errors of type conversions (reference manual, "Type conversions",
% "Mode checking"), with mmc's reasons: a ground value of a supertype,
% at the top or in an argument, which is not known to be one of the
% subtype; a bound inst with a functor the subtype lacks, at the top or
% in an argument; an argument that nothing binds; a result bound to a
% value the conversion cannot give, a warning, as in mmc; and a unique
% argument used after, which the conversion shares, as in mmc.
% These are mmc's uniqueness rules (-fmmc-uniqueness); by default, a
% value that is copied, such as a number, is unique wherever it is
% (../uniqueness/copied.m).
%
% RUN: not merkc -fc1 -fdebug-dump-modes -fmmc-uniqueness %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module coerce_errors.
:- interface.

:- type fruit ---> apple ; lemon ; orange.
:- type citrus =< fruit ---> lemon ; orange.
:- type tree ---> leaf ; node(tree, fruit, tree).
:- type ctree =< tree ---> leaf ; node(ctree, citrus, ctree).
:- type pair ---> p(fruit, fruit).
:- type cpair =< pair ---> p(citrus, fruit).
:- type box(T) ---> box(T) ; none.

:- func ground_down(fruit) = citrus.
:- func tree_down(tree) = ctree.
:- func pair_down(pair) = cpair.
:- func functor_down(fruit) = citrus.
:- func argument_down(fruit) = cpair.
:- func box_down(box(fruit)) = box(citrus).
:- pred unbound(citrus::out) is det.
:- pred cannot_succeed(citrus::in) is semidet.
:- pred unique_used(citrus::di, fruit::uo) is det.

:- implementation.

:- pred d(citrus::di) is det.
d(_).

ground_down(X) = coerce(X).
% CHECK: [[@LINE-1]]:18: error: in the clause for ground_down(in) = out: mode error in the coerce expression: cannot convert the argument X from type coerce_errors.fruit to coerce_errors.citrus, because it has inst ground, and coerce_errors.fruit is not a subtype of coerce_errors.citrus

tree_down(X) = coerce(X).
% CHECK: [[@LINE-1]]:16: error: in the clause for tree_down(in) = out: mode error in the coerce expression: cannot convert the argument X from type coerce_errors.tree to coerce_errors.ctree, because it has inst ground, and coerce_errors.tree is not a subtype of coerce_errors.ctree

pair_down(X) = coerce(X).
% CHECK: [[@LINE-1]]:16: error: in the clause for pair_down(in) = out: mode error in the coerce expression: cannot convert the argument X from type coerce_errors.pair to coerce_errors.cpair, because it has inst ground, and coerce_errors.pair is not a subtype of coerce_errors.cpair

functor_down(_) = coerce(Y) :-
% CHECK: [[@LINE-1]]:19: error: in the clause for functor_down(in) = out: mode error in the coerce expression: cannot convert the argument Y from type coerce_errors.fruit to coerce_errors.citrus, because its inst includes the function symbol apple
    Y = apple.

argument_down(X) = coerce(Y) :-
% CHECK: [[@LINE-1]]:20: error: in the clause for argument_down(in) = out: mode error in the coerce expression: in the first argument of function symbol p: cannot convert the subterm from type coerce_errors.fruit to coerce_errors.citrus, because its inst includes the function symbol apple
    Y = p(apple, X).

box_down(X) = coerce(Y) :-
% CHECK: [[@LINE-1]]:15: error: in the clause for box_down(in) = out: mode error in the coerce expression: in the argument of function symbol box: cannot convert the subterm from type coerce_errors.fruit to coerce_errors.citrus, because its inst includes the function symbol apple
    Y = box(apple),
    X = none.

unbound(Z) :-
    Z = coerce(_ : fruit).
% CHECK: [[@LINE-1]]:9: error: in the clause for unbound(out): mode error in the coerce expression: its argument V_1 has inst free, but it must be bound

cannot_succeed(X) :-
    Y = apple,
    Y = coerce(X).
% CHECK: [[@LINE-1]]:9: warning: the unification of Y and the coerce expression cannot succeed: Y has inst unique(apple), and the coerce expression has inst bound(lemon ; orange)

unique_used(X, Y) :-
    Y = coerce(X),
    d(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for unique_used(di, uo): mode error in argument 1 of the call to pred coerce_errors.d/1: X has inst ground, but unique is expected
