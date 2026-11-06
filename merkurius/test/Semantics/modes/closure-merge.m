% A closure keeps its higher-order inst across an if-then-else or a
% disjunction whose branches give it insts that differ only in uniqueness,
% as in mmc: a lambda's is unique, and passing it to a call gives it the
% mode's final inst, which is shared, so a call of the closure, or a
% second call that expects its inst, after the branches is accepted.
% Higher-order insts that differ in more merge to ground, which cannot be
% called.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>%t.err >/dev/null
% RUN: FileCheck --implicit-check-not=error: %s < %t.err
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module closure_merge.
:- interface.
:- pred after_condition(int::out, int::out) is semidet.
:- pred after_branches(int::in, int::out) is semidet.
:- pred after_disjunction(int::in, int::out) is semidet.
:- pred differing(int::in, int::out) is det.
:- implementation.
:- import_module int.

:- pred ap(pred(int, int), int, int).
:- mode ap(pred(in, out) is semidet, in, out) is semidet.
ap(P, X, Y) :-
    P(X, Y).

after_condition(A, B) :-
    P = (pred(X::in, Y::out) is semidet :- X < 10, Y = X),
    ( if ap(P, 1, A0) then A = A0 else A = 0 ),
    ap(P, 2, B).

after_branches(N, B) :-
    P = (pred(X::in, Y::out) is semidet :- X < 10, Y = X),
    ( if N > 0 then ap(P, N, _) else true ),
    P(N, B).

after_disjunction(N, B) :-
    P = (pred(X::in, Y::out) is semidet :- X < 10, Y = X),
    ( N = 1, ap(P, N, _) ; N = 2 ),
    ap(P, N, B).

differing(N, B) :-
    ( if N > 0 then
        P = (pred(X::in, Y::out) is det :- Y = X)
    else
        P = (pred(X::in, Y::out) is cc_multi :- Y = X)
    ),
    P(N, B).
% CHECK: [[@LINE-1]]:5: error: in the clause for differing(in, out): mode error in argument 1 (the predicate term) of the higher-order predicate call: P has inst unique, but a higher-order pred inst of arity 2 is expected
