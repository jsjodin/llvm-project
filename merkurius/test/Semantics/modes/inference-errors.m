% Mode errors of inferred modes (Overton, "Precise and Expressive Mode
% Systems for Typed Logic Programming Languages", 2003, section 3.4.2): a
% call whose inferred mode has a mode error waits for other insts, and if
% none come, as in mmc with --infer-modes, it is the caller's error, with
% the inferred mode's error, at its predicate's first clause, after it;
% such a mode with a ground argument; and an error after a call whose
% mode is inferred, in its final insts.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module inference_errors.
:- interface.
:- import_module list.

:- pred unbound(int::out) is det.
:- pred two_unbound(int::out) is det.
:- pred ground_argument(int::in, int::out) is semidet.
:- pred partial(list(int)::out) is det.

:- implementation.
:- import_module int.

:- pred use(int::in) is det.
use(_).

unbound(Y) :-
    bad(Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for unbound(out): mode error in the call to pred inference_errors.bad/1: the argument Y has inst free, which matches none of its valid modes, because its inferred mode bad(out(...)) has a mode error
% CHECK: [[@LINE+18]]:1: because: in the clause for the inferred mode bad(out(...)): mode error in argument 1 of the call to pred inference_errors.use/1: Y has inst free, but ground is expected

two_unbound(Y) :-
    inc(_, Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for two_unbound(out): mode error in the call to pred inference_errors.inc/2: the arguments V_1, Y have insts free, free, which match none of its valid modes, because its inferred mode inc(out(...), out(...)) has a mode error
% CHECK: [[@LINE+17]]:1: because: in the clause for the inferred mode inc(out(...), out(...)): mode error in the call to func int.+/2: the arguments X, V_2 and the result Y have insts free, unique(1), free, which match none of its modes

ground_argument(X, Y) :-
    two(X, Y).
% CHECK: [[@LINE-1]]:5: error: in the clause for ground_argument(in, out): mode error in the call to pred inference_errors.two/2: the arguments X, Y have insts ground, free, which match none of its valid modes, because its inferred mode two((ground >> ...), out(...)) has a mode error
% CHECK: [[@LINE+15]]:1: because: in the clause for the inferred mode two((ground >> ...), out(...)): mode error in the unification of Z and W, which are both free

partial(L) :-
    L = [_ | _],
    fill(L),
% CHECK: [[@LINE-1]]:5: error: in the clause for partial(out): mode error: argument 1, L, did not get sufficiently instantiated: its final inst is unique('[|]'(unique(1), free)), but ground is expected
    true.

bad(Y) :-
    true,
    use(Y).

inc(X, Y) :-
    Y = X + 1.

two(X, Y) :-
    X = 1,
    Y = 2.
two(X, Y) :-
    X = 2,
    Z = W,
    Y = Z + W.

fill([1 | _]).
