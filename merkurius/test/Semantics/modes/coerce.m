% Type conversions that mode analysis accepts (reference manual, "Type
% conversions", "Mode checking"): a ground value of a subtype, which gets
% the bound inst of the subtype's constructors, down to a recursive type,
% where it is ground; a value of a supertype whose inst has only the
% subtype's constructors; a bound inst, which keeps its functors; a
% unique value, which stays unique; one that waits until its argument is
% bound; one whose result is bound already, which is a test; and an
% argument used after, which is shared.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module coerce.
:- interface.

:- type fruit ---> apple ; lemon ; orange.
:- type citrus =< fruit ---> lemon ; orange.
:- type tree ---> leaf ; node(tree, fruit, tree).
:- type ctree =< tree ---> leaf ; node(ctree, citrus, ctree).
:- type pair ---> p(fruit, fruit).
:- type cpair =< pair ---> p(citrus, fruit).
:- inst citrus for fruit/0 ---> lemon ; orange.

:- func up(citrus) = fruit.
:- func down(fruit) = citrus.
:- mode down(in(citrus)) = out is det.
:- func tree_up(ctree) = tree.
:- func pair_up(cpair) = pair.
:- func pair_down(fruit) = cpair.
:- func unique_up(citrus::di) = (fruit::uo) is det.
:- pred waits(citrus::out, fruit::out) is det.
:- pred bound(citrus::in) is semidet.
:- pred shared(citrus::in, fruit::out, citrus::out) is det.
:- func narrow(citrus::in) = (fruit::out(citrus)) is det.

:- implementation.

up(X) = coerce(X).
% CHECK-LABEL: func coerce.up/1 mode 0: up(in) = out is det
% CHECK: construct HeadVar__2 = coerce(X)
% CHECK: final: X ground, HeadVar__2 bound(lemon ; orange)

down(X) = coerce(X).
% CHECK-LABEL: func coerce.down/1 mode 0: down(in(coerce.citrus)) = out is det
% CHECK: final: X coerce.citrus, HeadVar__2 bound(lemon ; orange)

tree_up(X) = coerce(X).
% CHECK-LABEL: func coerce.tree_up/1 mode 0: tree_up(in) = out is det
% CHECK: final: X ground, HeadVar__2 bound(leaf ; node(ground, bound(lemon ; orange), ground))

pair_up(X) = coerce(X).
% CHECK-LABEL: func coerce.pair_up/1 mode 0: pair_up(in) = out is det
% CHECK: final: X ground, HeadVar__2 bound(p(bound(lemon ; orange), ground))

pair_down(X) = coerce(Y) :-
    Y = p(lemon, X) : pair.
% CHECK-LABEL: func coerce.pair_down/1 mode 0: pair_down(in) = out is det
% CHECK: final: X ground, HeadVar__2 unique(p(unique(lemon), ground))

unique_up(X) = coerce(X).
% CHECK-LABEL: func coerce.unique_up/1 mode 0: unique_up(di) = uo is det
% CHECK: final: X unique, HeadVar__2 unique(lemon ; orange)

waits(X, Y) :-
    Y = coerce(X),
    X = lemon.
% CHECK-LABEL: pred coerce.waits/2 mode 0: waits(out, out) is det
% CHECK: construct X = lemon
% CHECK-NEXT: construct Y = coerce(X)
% CHECK-NEXT: final: X bound(lemon), Y bound(lemon)

bound(X) :-
    Y = lemon : fruit,
    Y = coerce(X).
% CHECK-LABEL: pred coerce.bound/1 mode 0: bound(in) is semidet
% CHECK: test Y = coerce(X)

shared(X, Y, Z) :-
    Y = coerce(X),
    Z = X.
% CHECK-LABEL: pred coerce.shared/3 mode 0: shared(in, out, out) is det
% CHECK: final: X ground, Y bound(lemon ; orange), Z ground

narrow(X) = coerce(X).
% CHECK-LABEL: func coerce.narrow/1 mode 0: narrow(in) = out(coerce.citrus) is det
% CHECK: final: X ground, HeadVar__2 bound(lemon ; orange)
