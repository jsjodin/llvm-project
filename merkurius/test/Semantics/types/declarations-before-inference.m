% Errors in the module's declarations are reported also when inference is
% the first to use them, and a signature with the error type in it prints
% after the checker is done.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error: %s

:- module declarations_before_inference.
:- interface.
:- pred main(int::out) is det.
:- implementation.
:- import_module list.

main(0).
:- type t ---> f(U).
% CHECK: [[@LINE-1]]:18: error: type variable U is not a parameter of the type
:- type e(T) == list(U).
% CHECK: [[@LINE-1]]:22: error: type variable U is not a parameter of the type
:- pred q(int) : int.
% CHECK: [[@LINE-1]]:4: error: the type of q must be a pred type
q(_).
p(X) :- X = f(1), q(2).
r(X) :- _ = X : e(int).
:- pred s(e(int)::in) is det.
s(_).
% CHECK: clause at [[@LINE-1]]:1 of pred declarations_before_inference.s/1: pred declarations_before_inference.s(list.list(<error>))
