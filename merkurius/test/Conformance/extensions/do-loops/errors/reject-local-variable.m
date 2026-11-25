% DESIGN: Loops / Errors
% A variable that occurs only in loops' bodies is local to each iteration
% of each, so loops one after another have different variables of the
% same name: the second loop's S is not the first's, and is unbound. The
% error names the loop that the value was most likely meant to leave.
% ORACLE: none
%
% RUN: not merkc -fc1 -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
:- module reject_local_variable.
:- interface.
:- import_module list.
:- pred p(list(int)::in, list(int)::in, int::out) is det.
:- implementation.
:- import_module int.
p(Xs, Ys, Sum) :-
    ( foreach(X, Xs) do S = X ),
    ( foreach(Y, Ys), fromto(0, A0, A1, Sum) do A1 = A0 + Y + S ).
% CHECK: reject-local-variable.m:[[@LINE-1]]:54: error: in the clause for p(in, in, out): in the loop at line [[@LINE-1]]: S is local to each iteration of the loop at line [[@LINE-2]]; to give a value out of a loop, use fromto(First, In, Out, S) or a state variable
