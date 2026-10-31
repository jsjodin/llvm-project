% Mode inference stops after a pass that finds a mode error in a
% procedure whose modes are not inferred, and its errors are those of that
% pass, as in mmc: here a call of a function without a declaration, whose
% mode is the default one, as in mmc, with an argument that is free. The
% call of bad/1 made its mode in that pass, and the goals after it were
% not reached, so the mode's error is not reported.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module inference_stop.
:- interface.

:- pred function_argument(int::out) is det.
:- pred unbound(int::out) is det.

:- implementation.
:- import_module int.

:- pred use(int::in) is det.
use(_).

function_argument(Y) :-
    Y = twice(_).
% CHECK: [[@LINE-1]]:9: error: in the clause for function_argument(out): mode error in argument 1 of the call to func inference_stop.twice/1: V_1 has inst free, but ground is expected

unbound(Y) :-
    bad(Y).

twice(X) = X * 2.

bad(Y) :-
    use(Y).
