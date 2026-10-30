% A predicate whose inferred type grows in every pass reaches the limit on
% passes, 60 as in mmc, which reports it too ("Type inference iteration
% limit exceeded"). Its clauses are then checked without a signature, so
% that its last guess draws no more errors.
%
% RUN: not merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error %s
% RUN: not merkc -fc1 -fdebug-dump-types -ftype-inference-iteration-limit=3 %s 2>&1 | FileCheck --check-prefix=THREE %s

:- module inference_limit.
:- interface.
:- pred top is det.
:- implementation.
:- import_module list.

top :- p(1, _).
p(X, Y) :- p([X], Y0), Y = [Y0].
% CHECK: inference-limit.m:[[@LINE-1]]:1: error: type inference did not reach a fixpoint for p/2 in 60 passes; declare its type
% CHECK: clause at [[@LINE-2]]:1 of pred inference_limit.p/2{{$}}
% THREE: inference-limit.m:[[@LINE-3]]:1: error: type inference did not reach a fixpoint for p/2 in 3 passes; declare its type
