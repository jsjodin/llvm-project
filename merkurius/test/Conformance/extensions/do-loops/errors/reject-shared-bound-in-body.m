% DESIGN: Loops / Variables
% A variable of the body that also occurs after the loop is shared, so it
% must be bound before the loop; S, bound only in the body, is an error
% that says so.
% ORACLE: none
%
% RUN: not merkc -fc1 -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_shared_bound_in_body.
:- interface.
:- import_module list.
:- pred p(list(int)::in, int::out) is det.
:- implementation.
p(Xs, Sum) :-
    ( foreach(X, Xs) do S = X ),
% CHECK: reject-shared-bound-in-body.m:[[@LINE-1]]:7: error: in the clause for p(in, out): S is shared by the iterations of the loop at line [[@LINE-1]], since it occurs around it, so it must be bound before the loop; to give a value out of a loop, use fromto(First, In, Out, S) or a state variable
    Sum = S.
