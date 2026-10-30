% Types of one module may share a constructor (reference manual,
% "Discriminated unions"): each type's red is a symbol of its own, and the
% context chooses one, also for an undeclared predicate.
%
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s

:- module shared_constructors.
:- interface.
:- type colour ---> red ; green.
:- type light ---> red ; off.
:- pred p(light::out) is det.
:- implementation.

p(L) :- L = red, L = off.
% CHECK:      clause at [[@LINE-1]]:1 of pred shared_constructors.p/1: pred shared_constructors.p(shared_constructors.light)
% CHECK-NEXT:   L: shared_constructors.light
% CHECK-NEXT:   [[@LINE-3]]:13 red -> constructor shared_constructors.red/0
q(C) :- C = red, C = green.
% CHECK:      clause at [[@LINE-1]]:1 of pred shared_constructors.q/1: inferred pred shared_constructors.q(shared_constructors.colour)
