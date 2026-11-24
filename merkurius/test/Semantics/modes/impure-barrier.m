% No goal moves across an impure one (reference manual, "Impurity"): a goal
% that waits for a variable that only a later goal binds is reordered after
% it in pure code, but not across an impure goal between them, which is a
% mode error instead, as in mmc. A unification of a head variable, or of
% the clause's argument that stands for one, may wait past it, as in mmc,
% since an impure procedure could not bind its outputs otherwise; a
% function application there is a call, which may not.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module impure_barrier.
:- interface.
:- import_module list.
:- pred reordered(int::out) is det.
:- impure pred blocked(int::out) is det.
:- impure pred after(int::out) is det.
:- impure pred head_term(list(int)::out) is det.
:- impure pred head_var(list(int)::out) is det.
:- impure pred head_alias(int::out) is det.
:- impure pred clauses(int::in, list(int)::out) is semidet.
:- impure func head_call = int.
:- implementation.
:- import_module int.

:- impure pred tick is det.
:- pragma foreign_proc("C", tick, [will_not_call_mercury], "").
:- impure pred next(int::out) is det.
:- pragma foreign_proc("C", next(X::out), [will_not_call_mercury], "X = 1;").

reordered(Y) :-
    Y = X + 1,
    X = 1.

blocked(Y) :-
    Y = X + 1,
% CHECK: [[@LINE-1]]:9: error: in the clause for blocked(out): mode error in
    impure tick,
% CHECK: [[@LINE-1]]:5: because: the goal could not be reordered, because it was followed by this impure goal
    X = 1.

    % After the impure goal, goals are reordered as usual.
after(Y) :-
    impure tick,
    Y = X + 1,
    X = 1.

head_term([X]) :-
    impure next(X).

head_var(R) :-
    R = [X],
    impure next(X).

head_alias(R) :-
    R = X,
    impure next(X).

clauses(0, [X]) :-
    impure next(X).
clauses(1, [X]) :-
    impure next(X).

head_call = X + 1 :-
% CHECK: [[@LINE-1]]:13: error: in the clause for head_call = out: mode error in
    impure next(X).
% CHECK: [[@LINE-1]]:5: because: the goal could not be reordered, because it was followed by this impure goal
