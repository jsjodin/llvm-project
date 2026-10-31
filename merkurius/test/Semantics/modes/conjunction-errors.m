% Mode errors in conjunctions of unifications and calls (Overton, "Precise
% and Expressive Mode Systems for Typed Logic Programming Languages", 2003,
% chapter 3): two free variables that are both used again, a final inst
% that does not match the mode's, in mmc's three cases, a call or a
% construction that still waits when nothing else can run, and a call
% that matches none of the callee's modes. Only the first error of a
% procedure is reported, as by mmc: at the goal that waits, or, for the
% final insts, at the goal that ran last.
%
% RUN: not merkc -fc1 -fdebug-dump-modes %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=warning: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module conjunction_errors.
:- interface.
:- import_module list.

:- pred free_free(int::out, int::out) is det.
:- pred not_bound(int::out) is det.
:- pred partial(list(int)::out) is det.
:- pred too_instantiated(int::unused) is det.
:- pred shared(list(int)::uo, int::out) is det.
:- pred waits(int::out) is det.
:- pred no_mode(int::out) is det.
:- pred function(int::in, int::out) is det.
:- pred construction(list(int)::out, int::out) is det.
:- pred first_only(int::out) is det.
:- pred last_run(int::out) is det.

:- implementation.
:- import_module int.

:- pred w(int::in, int::out) is det.
w(X, X).
:- pred m(int, int).
:- mode m(in, out) is det.
:- mode m(out, in) is det.
m(X, X).

free_free(X, Y) :- X = Y.
% CHECK: [[@LINE-1]]:20: error: in the clause for free_free(out, out): mode error in the unification of X and Y, which are both free
not_bound(_).
% CHECK: [[@LINE-1]]:1: error: in the clause for not_bound(out): mode error: argument 1, HeadVar__1, did not get sufficiently instantiated: its final inst is free, but ground is expected
partial([_]).
% CHECK: [[@LINE-1]]:9: error: in the clause for partial(out): mode error: argument 1, HeadVar__1, did not get sufficiently instantiated: its final inst is unique('[|]'(free, unique([]))), but ground is expected
too_instantiated(1).
% CHECK: [[@LINE-1]]:18: error: in the clause for too_instantiated(unused): mode error: argument 1, HeadVar__1, became too instantiated: its final inst is unique(1), but free is expected
shared(X, Z) :- Y = 1, X = [Y], Z = Y.
% CHECK: [[@LINE-1]]:33: error: in the clause for shared(uo, out): mode error: argument 1, X, had the wrong inst: its final inst is unique('[|]'(bound(1), unique([]))), but unique is expected
waits(X) :-
    w(Y, X),
% CHECK: [[@LINE-1]]:5: error: in the clause for waits(out): mode error in argument 1 of the call to pred conjunction_errors.w/2: Y has inst free, but ground is expected
    w(X, Y).
no_mode(X) :-
    m(Y, Z),
% CHECK: [[@LINE-1]]:5: error: in the clause for no_mode(out): mode error in the call to pred conjunction_errors.m/2: the arguments Y, Z have insts free, free, which match none of its modes
    X = Y + Z.
function(Y, X) :-
    X = Z + Y,
% CHECK: [[@LINE-1]]:9: error: in the clause for function(in, out): mode error in the call to func int.+/2: the arguments Z, Y and the result X have insts free, ground, free, which match none of its modes
    Z = W.
construction(L, H) :-
    L = [H],
% CHECK: [[@LINE-1]]:9: error: in the clause for construction(out, out): mode error in the unification of L and '[|]'(H, V_2): L has inst free, and '[|]'(H, V_2) has inst unique('[|]'(free, unique([])))
    H = _.
first_only(X) :-
    Y = Z,
% CHECK: [[@LINE-1]]:5: error: in the clause for first_only(out): mode error in the unification of Y and Z, which are both free
    w(Y, X),
    w(Z, _).
last_run(X) :-
    w(Y, _),
% CHECK: [[@LINE-1]]:5: error: in the clause for last_run(out): mode error: argument 1, X, did not get sufficiently instantiated: its final inst is free, but ground is expected
    Y = 1.
