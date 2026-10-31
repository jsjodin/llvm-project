% Errors for the any inst in negated contexts (reference manual, "Solver
% types and negated contexts"): a negation or an if-then-else condition
% with a nonlocal whose inst has an any part, of a solver type or not,
% outside a purity promise; one inside a some scope, or bound in the
% negation; a nested negation, reported first, as in mmc; and, as in
% mmc, one in a predicate with a promise_pure pragma or declared impure,
% which do not count as promises; and a pred lambda expression, which is
% ground, with an any nonlocal.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module any_errors.
:- interface.

:- solver type st.
:- type box ---> box(st).

:- pred negation(st::ia) is semidet.
:- pred not_solver(int::ia) is semidet.
:- pred condition(st::ia, int::out) is det.
:- pred term(st::ia) is semidet.
:- pred some_scope(st::ia) is semidet.
:- pred bound_inside(st::ia) is semidet.
:- pred nested(st::ia, int::out) is det.
:- pred pragma_promise(st::ia) is semidet.
% CHECK: [[@LINE-1]]:9: warning: the `promise_pure' pragma is unnecessary, since pred any_errors.pragma_promise/1 is pure
:- impure pred impure_pred(st::ia) is semidet.
% CHECK: [[@LINE-1]]:16: warning: pred any_errors.impure_pred/1 is declared impure, but it is pure
:- pred ground_lambda(st::ia) is semidet.

:- implementation.

:- solver type st
    where representation is int, ground is ground, any is ground.

:- pred r(st::ia) is semidet.
r(_) :-
    semidet_true.
:- pred ri(int::ia) is semidet.
ri(_) :-
    semidet_true.
:- pred rb(box::ia) is semidet.
rb(_) :-
    semidet_true.

negation(X) :-
    not r(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for negation(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

not_solver(X) :-
    not ri(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for not_solver(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

condition(X, Y) :-
    ( if r(X) then Y = 1 else Y = 2 ).
% CHECK: [[@LINE-1]]:7: error: in the clause for condition(ia, out): purity error: the if-then-else must be inside a promise_purity scope, because nonlocal X has inst any and appears in its condition

term(X) :-
    B = box(X),
    not rb(B).
% CHECK: [[@LINE-1]]:5: error: in the clause for term(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal B has inst any and appears in its body

some_scope(X) :-
    some [Y] ( Y = X, not r(Y) ).
% CHECK: [[@LINE-1]]:23: error: in the clause for some_scope(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal Y has inst any and appears in its body

bound_inside(X) :-
    not ( Y = X, r(Y) ).
% CHECK: [[@LINE-1]]:5: error: in the clause for bound_inside(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

nested(X, Y) :-
    ( if not r(X) then Y = 1 else Y = 2 ).
% CHECK: [[@LINE-1]]:10: error: in the clause for nested(ia, out): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

:- pragma promise_pure(pragma_promise/1).
pragma_promise(X) :-
    not r(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for pragma_promise(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

impure_pred(X) :-
    not r(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for impure_pred(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

ground_lambda(X) :-
    P = ((pred) is semidet :- r(X)),
% CHECK: [[@LINE-1]]:10: error: in the clause for ground_lambda(ia): purity error: the lambda expression is ground, but these nonlocals have insts that contain any: X
    call(P).
