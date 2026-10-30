% With -fno-infer-types, a clause for a predicate or function without a
% declaration is an error, in mmc's words without --infer-types; its type
% is still inferred, so its callers are checked by it.
%
% RUN: not merkc -fc1 -fdebug-dump-types -fno-infer-types %s 2>&1 | FileCheck %s
% RUN: merkc -fc1 -fdebug-dump-types %s 2>&1 | FileCheck --implicit-check-not=error --check-prefix=INFER %s

:- module no_infer_types.
:- interface.
:- pred p(int::out) is det.
:- implementation.
:- import_module int.

p(X) :- X = double(2).
double(X) = X * 2.
% CHECK: no-infer-types.m:[[@LINE-1]]:1: error: clause for function double/1 without corresponding `:- func' declaration
% CHECK: clause at [[@LINE-2]]:1 of func no_infer_types.double/1: inferred func no_infer_types.double(int) = int
% INFER: clause at [[@LINE-3]]:1 of func no_infer_types.double/1: inferred func no_infer_types.double(int) = int
