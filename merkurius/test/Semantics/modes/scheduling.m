% How mode analysis schedules a conjunction, as mmc does: a waiting goal
% is woken when a variable it waits for changes, and runs before the
% others; a call with one mode waits only for its first argument that
% does not match, so the error reported is that of the first goal to
% wait. A value with free parts may not be copied while both copies are
% used again, since mode analysis does not follow aliases, but may once
% one of them is not. Aliasing a variable to one not used again wakes the
% goals that wait for it, which may now find it dead. After a goal that
% cannot succeed, the rest of the conjunction is not reached, and
% mode-correct, with a warning.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module scheduling.
:- interface.
:- import_module list.

:- pred first_waiting(int::out) is det.
:- pred partial_alias(list(int)::out) is det.
:- pred partial_argument(list(list(int))::out) is det.
:- pred dead_alias(list(int)::out) is det.
:- pred unreachable(list(int)::out) is failure.

:- implementation.
:- import_module int.

:- pred use(int::in) is det.
use(_).
:- pred q(int::in, int::out) is det.
q(X, X).

first_waiting(X) :-
    q(A, B),
% CHECK: [[@LINE-1]]:5: error: in the clause for first_waiting(out): mode error in argument 1 of the call to pred scheduling.q/2: A has inst free, but ground is expected
    use(C),
    use(D),
    C = D,
    B = 1,
    X = A.
% N = M waits until M = [1] has bound M's free parts, since a copy made
% before would not see them.
partial_alias(L) :-
    M = [_ | _],
    N = M,
    M = [1],
    L = N.
partial_argument(L) :-
    M = [_ | _],
    L = [M],
    M = [1].
dead_alias(L) :-
    M = [1 | T],
    _ = M,
    T = [2 | U],
    U = [].
% CHECK: [[@LINE-1]]:9: error: in the clause for dead_alias(out): mode error: argument 1, L, did not get sufficiently instantiated: its final inst is free, but ground is expected
unreachable(L) :-
    L = [],
    L = [_ | _],
% CHECK: [[@LINE-1]]:9: warning: the unification of L and '[|]'(V_1, V_2) cannot succeed: L has inst unique([])
    use(_).
