% Instance methods that mode analysis accepts: an instance's clauses for a
% method, and a predicate or function it names, are checked in each of the
% method's modes, those of the type class's declarations, as mmc does.
% The clauses of a method with two modes are checked in both.
%
% RUN: merkc -fc1 -fdebug-dump-modes %s | FileCheck %s
% RUN: %if mmc %{ %compare-types --modes --error-lines %s %}

:- module methods.
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
:- instance c(a).
:- instance c(b).

:- implementation.

:- instance c(a) where [
    m(X, X),
    f(X) = X,
    (u(X, Y) :- Y = X),
    two(X, X)
].
% CHECK-LABEL: method methods.c(methods.a) pred methods.m/2 mode 0
% CHECK: assign HeadVar__2 := X
% CHECK-LABEL: method methods.c(methods.a) func methods.f/1 mode 0
% CHECK-LABEL: method methods.c(methods.a) pred methods.u/2 mode 0
% CHECK: final: X unique, Y unique
% CHECK-LABEL: method methods.c(methods.a) pred methods.two/2 mode 0
% CHECK: assign HeadVar__2 := X
% CHECK-LABEL: method methods.c(methods.a) pred methods.two/2 mode 1
% CHECK: assign X := HeadVar__2

:- instance c(b) where [
    pred(m/2) is mb,
    func(f/1) is fb,
    pred(u/2) is ub,
    pred(two/2) is twob
].
% CHECK-LABEL: method methods.c(methods.b) pred methods.m/2 mode 0
% CHECK: call pred methods.mb/2 mode 0: mb(HeadVar__1, HeadVar__2)
% CHECK-LABEL: method methods.c(methods.b) func methods.f/1 mode 0
% CHECK: call func methods.fb/1 mode 0: HeadVar__2 = fb(HeadVar__1)
% CHECK-LABEL: method methods.c(methods.b) pred methods.u/2 mode 0
% CHECK: call pred methods.ub/2 mode 0: ub(HeadVar__1, HeadVar__2)
% CHECK-LABEL: method methods.c(methods.b) pred methods.two/2 mode 0
% CHECK: call pred methods.twob/2 mode 0: twob(HeadVar__1, HeadVar__2)
% CHECK-LABEL: method methods.c(methods.b) pred methods.two/2 mode 1
% CHECK: call pred methods.twob/2 mode 1: twob(HeadVar__1, HeadVar__2)

:- pred mb(b::in, b::out) is det.
mb(X, X).
:- func fb(b) = b.
fb(X) = X.
:- pred ub(b::di, b::uo) is det.
ub(X, X).
:- pred twob(b, b).
:- mode twob(in, out) is det.
:- mode twob(out, in) is det.
twob(X, X).
