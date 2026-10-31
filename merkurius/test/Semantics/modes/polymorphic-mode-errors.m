% Mode errors with constrained polymorphic modes (reference manual,
% "Constrained polymorphic modes"; Overton, "Precise and Expressive Mode
% Systems for Typed Logic Programming Languages", 2003, section 4.2): a
% final inst without the inst variable, as for a value bound in the
% clause, one of another inst variable, or a merge of branches where only
% one has it, which mmc says "became too instantiated"; an argument that
% does not match the constraint; an argument that does not match what an
% earlier argument bound the inst variable to, as mmc binds it where it
% first occurs; an inst variable's inst, narrowed or not, passed where
% another inst is expected; a higher-order term of a predicate with an
% inst variable in its modes whose inputs bind it, which mmc rejects as
% the lambda expression it makes; and, as in mmc, a higher-order call
% whose inst variable is the caller's own of the same number, and a
% narrowed inst variable passed where the callee has one, both rejected
% (see the design document's register).
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module polymorphic_mode_errors.
:- interface.
:- import_module list.

:- type t ---> a ; b ; c ; f(int).

:- pred bound_here(t::in(I), t::out(I)) is det.
:- pred not_bound(t::in(I), t::out(I)) is det.
:- pred other_variable(t::in(I), t::in(J), t::out(J)) is det.
:- pred one_branch(t::in(I), t::out(I)) is semidet.
:- pred constraint(t::out) is det.
:- pred first_wins(list(t)::out) is det.
:- pred narrowed(t::in(I)) is semidet.
:- pred both(t::in(I), t::in(J)) is semidet.
:- pred curried(t::out) is det.
:- pred own_number(int::in(J)) is det.
:- pred narrowed_call(t::in(I), t::out) is semidet.

:- implementation.

:- pred id(T::in(I), T::out(I)) is det.
id(X, X).
:- pred ab(t::in(I =< bound(a ; b)), t::out(I =< bound(a ; b))) is det.
ab(X, X).
:- pred app(list(T), list(T), list(T)).
:- mode app(in(list_skel(I)), in(list_skel(I)), out(list_skel(I))) is det.
app([], Ys, Ys).
app([X | Xs], Ys, [X | Zs]) :- app(Xs, Ys, Zs).
:- pred need_b(t::in(bound(b))) is det.
need_b(_).
:- pred p0(int::in(I), int::in) is det.
p0(_, _).

bound_here(_, a).
% CHECK: [[@LINE-1]]:15: error: in the clause for bound_here(in((I =< ground)), out((I =< ground))): mode error: argument 2, HeadVar__2, became too instantiated: its final inst is unique(a), but (I =< ground) is expected
not_bound(_, _).
% CHECK: [[@LINE-1]]:1: error: in the clause for not_bound(in((I =< ground)), out((I =< ground))): mode error: argument 2, HeadVar__2, did not get sufficiently instantiated: its final inst is free, but (I =< ground) is expected
other_variable(X, _, Z) :-
    Z = X.
% CHECK: [[@LINE-1]]:5: error: in the clause for other_variable(in((I =< ground)), in((J =< ground)), out((J =< ground))): mode error: argument 3, Z, became too instantiated: its final inst is (I =< ground), but (J =< ground) is expected
one_branch(X, Y) :-
    (
        X = a,
        Y = X
    ;
        Y = b
% CHECK: [[@LINE-1]]:13: error: in the clause for one_branch(in((I =< ground)), out((I =< ground))): mode error: argument 2, Y, became too instantiated: its final inst is bound(a ; b), but (I =< ground) is expected
    ).
constraint(Y) :-
    X = c,
    ab(X, Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for constraint(out): mode error in argument 1 of the call to pred polymorphic_mode_errors.ab/2: X has inst unique(c), but (I =< bound(a ; b)) is expected
first_wins(L) :-
    A = a,
    B = b,
    app([A], [B], L).
% CHECK: [[@LINE-1]]:5: error: in the clause for first_wins(out): mode error in argument 2 of the call to pred polymorphic_mode_errors.app/3: V_5 has inst unique('[|]'(unique(b), unique([]))), but list.list_skel(bound(a)) is expected
narrowed(X) :-
    X = a,
    need_b(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for narrowed(in((I =< ground))): mode error in argument 1 of the call to pred polymorphic_mode_errors.need_b/1: X has inst (I =< bound(a)), but bound(b) is expected
both(X, Y) :-
    X = Y,
    need_b(X).
% CHECK: [[@LINE-1]]:5: error: in the clause for both(in((I =< ground)), in((J =< ground))): mode error in argument 1 of the call to pred polymorphic_mode_errors.need_b/1: X has inst (I, J =< ground), but bound(b) is expected
curried(Y) :-
    A = a,
    P = id(A),
% CHECK: [[@LINE-1]]:9: error: in the clause for curried(out): mode error: argument 1 of the higher-order term became too instantiated: its final inst is bound(a), but (I =< ground) is expected
    P(Y).
own_number(X) :-
    P = p0,
    A = 1,
    P(A, X).
% CHECK: [[@LINE-1]]:5: error: in the clause for own_number(in((J =< ground))): mode error in argument 2 (argument 1 of the called predicate) of the higher-order predicate call: A has inst unique(1), but (J =< ground) is expected
narrowed_call(X, Y) :-
    X = a,
    id(X, Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for narrowed_call(in((I =< ground)), out): mode error in argument 1 of the call to pred polymorphic_mode_errors.id/2: X has inst (I =< bound(a)), but (I =< ground) is expected
