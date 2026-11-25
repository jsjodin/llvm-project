% DESIGN: Loops / Variables
% A variable of the body that also occurs around the loop, here in the
% clause's head, is shared by its iterations, so it must be bound before
% the loop: a value leaves a loop through an iterator or a state variable.
% ORACLE: none
%
% RUN: not merkc -fc1 -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_shared_unbound.
:- interface.
:- import_module list.
:- pred p(list(int)::in, int::out) is det.
:- implementation.
p(Xs, Sum) :- ( foreach(X, Xs) do Sum = X ).
% CHECK: reject-shared-unbound.m:[[@LINE-1]]:17: error: in the clause for p(in, out): Sum is shared by the iterations of the loop at line [[@LINE-1]], since it occurs around it, so it must be bound before the loop; to give a value out of a loop, use fromto(First, In, Out, Sum) or a state variable
