% The any inst that mode analysis accepts (reference manual, "Solver
% types"): an ia argument passed on and an oa output assigned; a
% negation or a condition whose any nonlocals are inside a purity
% promise; a ground solver value in a negation; an any_pred lambda
% expression, whose nonlocals may be any; and, as in mmc, a negated call
% of such a closure, though the manual
% asks for a promise there too (see the design document's register).
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module any.
:- interface.

:- solver type st.

:- pred passed(st::ia, st::oa) is semidet.
:- pred promised(st::ia) is semidet.
:- pred promised_condition(st::ia, int::out) is det.
:- pred ground_value(st::in) is semidet.
:- pred any_lambda(st::ia) is semidet.
:- pred negated_closure(st::ia) is semidet.

:- implementation.

:- solver type st
    where representation is int, ground is ground, any is ground.

:- pred r(st::ia) is semidet.
r(_) :-
    semidet_true.
:- pred rg(st::in) is semidet.
rg(_) :-
    semidet_true.

passed(X, Y) :-
    r(X),
    Y = X.
% CHECK-LABEL: pred any.passed/2 mode 0: passed(ia, oa) is semidet
% CHECK: final: X any, Y any

promised(X) :-
    promise_pure ( not r(X) ).
% CHECK-LABEL: pred any.promised/1 mode 0: promised(ia) is semidet
% CHECK: not

promised_condition(X, Y) :-
    promise_pure ( if r(X) then Y = 1 else Y = 2 ).
% CHECK-LABEL: pred any.promised_condition/2 mode 0: promised_condition(ia, out) is det
% CHECK: final: X any, Y unique(1 ; 2)

ground_value(X) :-
    not rg(X).
% CHECK-LABEL: pred any.ground_value/1 mode 0: ground_value(in) is semidet
% CHECK: final: X ground

any_lambda(X) :-
    P = (any_pred is semidet :- r(X)),
    P.
% CHECK-LABEL: pred any.any_lambda/1 mode 0: any_lambda(ia) is semidet
% CHECK: construct P = lambda expression
% CHECK: higher-order call: P()

negated_closure(X) :-
    P = (any_pred is semidet :- r(X)),
    not P.
% CHECK-LABEL: pred any.negated_closure/1 mode 0: negated_closure(ia) is semidet
% CHECK: not
% CHECK-NEXT: higher-order call: P()
