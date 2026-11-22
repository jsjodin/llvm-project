% Mode and determinism errors of open terms (design document, "Open terms as
% built"): binding an existing variable may fail, and so may a get_structure of
% a type with several functors, and of closed(T), which may be unbound;
% closed(T) holds a ground value; a condition that may bind an open nonlocal
% needs a purity promise; and free does not match any, so a branch may not leave
% free what another binds; and an open term is never unique.
%
% RUN: not merkc -fc1 -fopen-types -fdebug-dump-determinism %s 2>&1 | FileCheck --implicit-check-not=error: %s

:- module open_mode_errors.
:- interface.
:- import_module open, list.
:- pred nil(open(list(int))::ia) is det.
:- pred same(open(int)::ia, open(int)::ia) is det.
:- pred c(int::ia, closed(int)::out) is det.
:- pred neg(open(int)::ia) is semidet.
:- pred merge(int::in, open(int)::oa) is det.
:- implementation.
:- import_module int.

nil(X) :- X = [].
% CHECK: [[@LINE-9]]:9: error: the determinism declaration of nil(ia) is not satisfied: it is declared det, but inferred semidet
% CHECK: [[@LINE-2]]:15: because: the unification of X and []/0 can fail

same(X, X).
% CHECK: [[@LINE-12]]:9: error: the determinism declaration of same(ia, ia) is not satisfied: it is declared det, but inferred semidet
% CHECK: [[@LINE-2]]:9: because: the unification of HeadVar__2 and X can fail

c(X, closed(X)).
% CHECK: [[@LINE-1]]:6: error: in the clause for c(ia, out): mode error in the unification of HeadVar__2 and closed(X): closed(T) holds a ground value, but X has inst any

neg(X) :- \+ X = to_open(1).
% CHECK: [[@LINE-1]]:11: error: in the clause for neg(ia): purity error: the negation must be inside a promise_purity scope, because nonlocal X has inst any and appears in its body

merge(N, X) :- ( if N > 0 then X = to_open(1) else true ).
% CHECK: [[@LINE-1]]:52: error: in the clause for merge(in, oa): mode mismatch in the if-then-else: X is ground in some branches but not others
% CHECK: [[@LINE-2]]:32: because: in this branch, X has inst ground
% CHECK: [[@LINE-3]]:52: because: in this branch, X has inst free

    % An open term is never unique.
:- pred mk(open(list(int))::uo) is det.
mk([_]).
% CHECK: [[@LINE-1]]:4: error: in the clause for mk(uo): mode error: argument 1, HeadVar__1, had the wrong inst: its final inst is bound('[|]'(any, bound([]))), but unique is expected

    % closed(T) inside an open term may be unbound, so taking its argument
    % may fail: C below is still a fresh variable when unclose runs.
:- type bundle ---> bundle(closed(list(int)), int).
:- pred unclose(closed(list(int))::ia, list(int)::out) is det.
unclose(closed(L), L).
% CHECK: [[@LINE-2]]:9: error: the determinism declaration of unclose(ia, out) is not satisfied: it is declared det, but inferred semidet
% CHECK: [[@LINE-2]]:9: because: the unification of HeadVar__1 and closed/1 can fail

    % A disjunction that may bind variables inside a nonlocal keeps its
    % solutions: by the nonlocal's inst, which has `any` in it, whatever the
    % depth of the open type, or through a type variable that may be
    % open(U) at the call.
:- pred bind(list(open(int))::ia, int::in) is semidet.
bind(L, N) :- L = [Z], Z = to_open(N).
:- pred nested(list(open(int))::ia) is semidet.
nested(L) :- promise_pure ( ( bind(L, 1) ; bind(L, 2) ), bind(L, 2) ).
% CHECK: [[@LINE-2]]:9: error: the determinism declaration of nested(ia) is not satisfied: it is declared semidet, but inferred nondet
% CHECK: [[@LINE-2]]:31: because: the disjunction has more than one disjunct with solutions
:- pred hoisted(list(open(int))::ia) is semidet.
hoisted(L) :- promise_pure (
    ( L = [Z], Z = to_open(1) ; L = [Z], Z = to_open(2) ),
    L = [W], W = to_open(2) ).
% CHECK: [[@LINE-4]]:9: error: the determinism declaration of hoisted(ia) is not satisfied: it is declared semidet, but inferred nondet
% CHECK: [[@LINE-3]]:7: because: the disjunction has more than one disjunct with solutions
:- pred poly(T::ia, pred(T, int)::in(pred(ia, in) is semidet)) is semidet.
poly(X, R) :- promise_pure ( ( R(X, 1) ; R(X, 2) ), R(X, 2) ).
% CHECK: [[@LINE-2]]:9: error: the determinism declaration of poly(ia, in((pred(ia, in) is semidet))) is not satisfied: it is declared semidet, but inferred nondet
% CHECK: [[@LINE-2]]:32: because: the disjunction has more than one disjunct with solutions
