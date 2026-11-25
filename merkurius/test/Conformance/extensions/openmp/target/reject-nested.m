% DESIGN: Target regions
% The directive belongs to the caller: a target region in a procedure that
% a target region calls would run on the device, as reverse offloading
% would, so it is an error, which names the procedure.
% ORACLE: none
%
% RUN: not merkc -fopenmp -c %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_nested.
:- interface.
:- pred outer(int::in, int::out) is det.
:- implementation.
:- import_module int.

:- pred inner(int::in, int::out) is det.
inner(X, Y) :-
    %$omp target
    Y = X + 1.
% CHECK: reject-nested.m:[[@LINE-2]]:5: error: a target region cannot be in reject_nested.inner_2_0, which a target region calls

outer(X, Y) :-
    %$omp target
    inner(X, Y).
