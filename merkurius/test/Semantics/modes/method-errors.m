% Mode errors in instance methods, reported for the method's mode, as mmc
% does: a clause whose output is left free; a named predicate whose modes
% do not fit the method's, where mmc reports the argument that an implied
% mode cannot fix; a named predicate that gives a ground output for a uo
% one; a method with two modes, of which the named predicate has only
% one; a clause that clobbers a variable used after; and one whose uo
% output is shared.
% These are mmc's uniqueness rules (-fmmc-uniqueness); by default, a
% value that is copied, such as a number, is unique wherever it is
% (../uniqueness/copied.m).
%
% RUN: not merkc -fc1 -fdebug-dump-modes -fmmc-uniqueness %s 2>&1 >/dev/null | FileCheck --implicit-check-not=error: --implicit-check-not=because: %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module method_errors.
:- interface.

:- typeclass c(T) where [
    pred m(T::in, T::out) is det,
    func f(T) = T,
    pred u(T::di, T::uo) is det,
    pred two(T, T),
    mode two(in, out) is det,
    mode two(out, in) is det
].

:- type a ---> a.
:- type b ---> b.
:- type d ---> d.
:- instance c(a).
:- instance c(b).
:- instance c(d).

:- implementation.

:- instance c(a) where [
    m(X, X),
    f(_) = _,
% CHECK: [[@LINE-1]]:5: error: in the clause for the method f(in) = out of instance method_errors.c(method_errors.a): mode error: argument 2, HeadVar__2, did not get sufficiently instantiated: its final inst is free, but ground is expected
    (u(X, Y) :- Y = X),
    two(X, X)
].

:- instance c(b) where [
    pred(m/2) is mb,
% CHECK: [[@LINE-1]]:18: error: in the clause for the method m(in, out) of instance method_errors.c(method_errors.b): mode error in argument 2 of the call to pred method_errors.mb/2: HeadVar__2 has inst free, but ground is expected
    func(f/1) is fb,
    pred(u/2) is ub,
% CHECK: [[@LINE-1]]:18: error: in the clause for the method u(di, uo) of instance method_errors.c(method_errors.b): mode error: argument 2, HeadVar__2, did not get sufficiently instantiated: its final inst is ground, but unique is expected
    pred(two/2) is twob
% CHECK: [[@LINE-1]]:20: error: in the clause for the method two(out, in) of instance method_errors.c(method_errors.b): mode error in argument 1 of the call to pred method_errors.twob/2: HeadVar__1 has inst free, but ground is expected
].

:- instance c(d) where [
    (m(X, Y) :- d(X), Y = X),
% CHECK: [[@LINE-1]]:17: error: in the clause for the method m(in, out) of instance method_errors.c(method_errors.d): unique-mode error in argument 1 of the call to pred method_errors.d/1: the called procedure would clobber its argument, but X is still live
    f(X) = X,
    (u(X, Y) :- Y = X, use(X)),
% CHECK: [[@LINE-1]]:24: error: in the clause for the method u(di, uo) of instance method_errors.c(method_errors.d): mode error: argument 2, Y, did not get sufficiently instantiated: its final inst is ground, but unique is expected
    (two(X, Y) :- Y = X)
].

:- pred mb(b::out, b::in) is det.
mb(b, _).
:- func fb(b::in) = (b::out) is det.
fb(X) = X.
:- pred ub(b::in, b::out) is det.
ub(X, X).
:- pred twob(b::in, b::out) is det.
twob(X, X).
:- pred d(d::di) is det.
d(_).
:- pred use(d::in) is det.
use(_).
