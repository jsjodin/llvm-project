% Mode errors of higher-order terms and calls (reference manual,
% "Higher-order"; Overton, "Precise and Expressive Mode Systems for Typed
% Logic Programming Languages", 2003, 3.2.5): a call of a term without a
% higher-order inst, or with an argument that does not fit; a term of a
% predicate with several modes whose higher-order arguments match more
% than one, or none; a closure that does not match the inst expected; a
% unification of two higher-order terms, also as arguments of two terms,
% where a deconstruction tests a bound argument against a bound field,
% though not through a variable of the terms' type, which mmc leaves to
% run time as the reference manual allows ("Higher-order"); a function that does not have
% the default inst passed for ground; a lambda expression's nonlocal or a
% curried argument that is not ground; and a lambda expression whose
% body has a mode error, or whose argument does not get its final inst,
% which waits for no variable, and so comes first after a goal in which
% something cannot succeed wakes every goal, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module higher_order_errors.
:- interface.
:- import_module list.

:- pred ground_call(pred(int)::in) is det.
:- pred free_closure(int::out) is det.
:- pred free_argument(int::out) is det.
:- pred function_argument(int::out) is det.
:- pred ambiguous(int::out) is det.
:- pred matches_none(int::out) is det.
:- pred match_none(int::out) is det.
:- pred semidet_closure is det.
:- pred two_closures is semidet.
:- pred bound_closure(pred(int)::in) is semidet.
:- pred not_default is det.
:- pred nonlocal(int::out) is det.
:- pred curried(int::out) is det.
:- pred curried_misfit(int::out) is det.
:- pred body(int::out) is det.
:- pred final(int::out) is det.
:- pred wrong_final(pred(int)::out(pred(out) is det)) is det.
:- pred from_list(list(pred(int))::in, int::out) is semidet.
:- pred woken(int::out) is semidet.
:- pred in_terms(pred(int)::in, pred(int)::in) is semidet.
:- pred in_head(pred(int)::in, list(pred(int))::in) is semidet.
:- pred indirect(pred(int)::in, pred(int)::in) is semidet.

:- implementation.
:- import_module int.

:- pred w(int::in, int::out) is det.
w(X, X).
:- pred r(int::out, int::in) is det.
r(X, X).
:- pred m(int, int).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
m(X, X).
:- pred app(pred(int), int).
:- mode app(in(pred(in) is det), in) is det.
:- mode app(in(pred(out) is det), out) is det.
app(P, X) :- P(X).
:- pred app2(pred(int), pred(int), int).
:- mode app2(in(pred(in) is det), in(pred(in) is det), in) is det.
:- mode app2(in(pred(out) is det), in(pred(out) is det), out) is det.
app2(P, _, X) :- P(X).
:- func plus1(int, int) = int.
plus1(X, Y) = X + Y.
:- pred takes(pred(int)::in(pred(in) is det)) is det.
takes(_).
:- pred takes_function((func(int) = int)::in) is det.
takes_function(_).
:- pred s(int::in) is semidet.
s(1).
:- pred use(int::in) is det.
use(_).

ground_call(P) :-
    P(1).
% CHECK: [[@LINE-1]]:5: error: in the clause for ground_call(in): mode error in argument 1 (the predicate term) of the higher-order predicate call: P has inst ground, but a higher-order pred inst of arity 1 is expected
free_closure(Y) :-
    P(Y),
% CHECK: [[@LINE-1]]:5: error: in the clause for free_closure(out): mode error in argument 1 (the predicate term) of the higher-order predicate call: P has inst free, but a higher-order pred inst of arity 1 is expected
    _ = P.
free_argument(Y) :-
    P = r(1),
    P(Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for free_argument(out): mode error in argument 2 (argument 1 of the called predicate) of the higher-order predicate call: Y has inst free, but ground is expected
function_argument(Y) :-
    F = plus1(1),
    Y = F(_).
% CHECK: [[@LINE-1]]:9: error: in the clause for function_argument(out): mode error in argument 2 (argument 1 of the called function) of the higher-order function call: V_3 has inst free, but ground is expected
ambiguous(Y) :-
    P = m(1),
% CHECK: [[@LINE-1]]:9: error: in the clause for ambiguous(out): mode error in the unification of P and a higher-order term based on the multi-moded pred higher_order_errors.m/2: the modes of the argument variables match more than one of its modes
    P(Y).
matches_none(Y) :-
    P = app2(_, _),
% CHECK: [[@LINE-1]]:9: error: in the clause for matches_none(out): mode error in the unification of P and a higher-order term based on the multi-moded pred higher_order_errors.app2/3: the modes of the argument variables do not match any of its modes
    P(Y).
match_none(Y) :-
    P = app(_),
% CHECK: [[@LINE-1]]:9: error: in the clause for match_none(out): mode error in the unification of P and a higher-order term based on the multi-moded pred higher_order_errors.app/2: the mode of the argument variable does not match any of its modes
    P(Y).
semidet_closure :-
    takes(s).
% CHECK: [[@LINE-1]]:5: error: in the clause for semidet_closure: mode error in argument 1 of the call to pred higher_order_errors.takes/1: V_0 has inst /* unique */ (pred(in) is semidet), but (pred(in) is det) is expected
two_closures :-
    P = w(1),
    Q = w(2),
    P = Q.
% CHECK: [[@LINE-1]]:5: error: in the clause for two_closures: mode error in the unification of P and Q: attempt at higher-order unification: two terms of type pred(int) cannot be unified
bound_closure(P) :-
    P = (pred(X::in) is semidet :- X = 1).
% CHECK: [[@LINE-1]]:10: error: in the clause for bound_closure(in): mode error in the unification of P and a lambda expression: attempt at higher-order unification: two terms of type pred(int) cannot be unified
not_default :-
    F = (func(X::in) = (Y::out) is semidet :- X = 1, Y = 2),
    takes_function(F).
% CHECK: [[@LINE-1]]:5: error: in the clause for not_default: mode error in argument 1 of the call to pred higher_order_errors.takes_function/1: F has inst /* unique */ (func(in) = out is semidet), but (func(in) = out is det) is expected
nonlocal(Z) :-
    P = (pred(X::out) is det :- X = Z),
% CHECK: [[@LINE-1]]:10: error: in the clause for nonlocal(out): mode error: Z has inst free, but the nonlocal variables of a lambda expression must be ground
    P(_).
curried(Y) :-
    P = r(V),
% CHECK: [[@LINE-1]]:9: error: in the clause for curried(out): mode error: V has inst free, but the nonlocal variables of a lambda expression must be ground
    P(1),
    use(V),
    Y = 1.
curried_misfit(Y) :-
    P = w(_),
% CHECK: [[@LINE-1]]:9: error: in the clause for curried_misfit(out): mode error in argument 1 of the call to pred higher_order_errors.w/2: V_2 has inst free, but ground is expected
    P(Y).
body(Y) :-
    P = (pred(X::out) is det :-
        w(V, X)
% CHECK: [[@LINE-1]]:9: error: in the clause for body(out): mode error in argument 1 of the call to pred higher_order_errors.w/2: V has inst free, but ground is expected
    ),
    P(Y).
final(Y) :-
    P = (pred(X::out) is det :-
% CHECK: [[@LINE-1]]:10: error: in the clause for final(out): mode error: argument 1, LambdaHeadVar__1, did not get sufficiently instantiated: its final inst is free, but ground is expected
        true
    ),
    P(Y),
    use(Y).
wrong_final(P) :-
    P = (pred(X::out) is semidet :- X = 1).
% CHECK: [[@LINE-1]]:10: error: in the clause for wrong_final(out((pred(out) is det))): mode error: argument 1, P, had the wrong inst: its final inst is /* unique */ (pred(out) is semidet), but (pred(out) is det) is expected
from_list(L, Y) :-
    L = [P | _],
    P(Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for from_list(in, out): mode error in argument 1 (the predicate term) of the higher-order predicate call: P has inst ground, but a higher-order pred inst of arity 1 is expected
woken(Y) :-
    use(V7),
    not (V1 = 5),
% CHECK: [[@LINE-1]]:15: warning: the unification of V1 and 5 cannot succeed: V1 has inst unique(3)
    V5(V7),
    V5 = (pred(_V6::out) is det :- true),
% CHECK: [[@LINE-1]]:11: error: in the clause for woken(out): mode error: argument 1, LambdaHeadVar__1, did not get sufficiently instantiated: its final inst is free, but ground is expected
    V1 = 3,
    Y = 1.
in_terms(P, Q) :-
    [P] = [Q].
% CHECK: [[@LINE-1]]:11: error: in the clause for in_terms(in, in): mode error in the unification of Q and argument 1 of V_2: attempt at higher-order unification: two terms of type pred(int) cannot be unified
in_head(P, L) :-
    L = [P | _].
% CHECK: [[@LINE-1]]:9: error: in the clause for in_head(in, in): mode error in the unification of P and argument 1 of L: attempt at higher-order unification: two terms of type pred(int) cannot be unified
indirect(P, Q) :-
    X = [P],
    Y = [Q],
    X = Y.
