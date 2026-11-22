% No goal moves across an impure one (reference manual, "Impurity"): a goal
% that waits for a variable that only a later goal binds is reordered after
% it in pure code, but not across an impure goal between them, which is a
% mode error instead, as in mmc.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module impure_barrier.
:- interface.
:- pred reordered(int::out) is det.
:- impure pred blocked(int::out) is det.
:- impure pred after(int::out) is det.
:- implementation.
:- import_module int.

:- impure pred tick is det.
:- pragma foreign_proc("C", tick, [will_not_call_mercury], "").

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
