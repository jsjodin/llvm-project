% DESIGN: Target regions / Data regions
% A target data region's goal must be det, since failing would leave the
% region without unmapping its arrays. A state variable that the goal
% updates needs a map type that copies it back, and a map names a variable
% or state variable that the goal uses, once.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_data.
:- interface.
:- import_module array.
:- pred semi(array(int)::array_ui, int::out) is semidet.
:- pred copied_to(array(int)::array_di, array(int)::array_uo) is det.
:- pred unused(array(int)::array_ui, array(int)::array_ui, int::out) is det.
:- pred twice(array(int)::array_ui, int::out) is det.
:- implementation.
:- import_module int.

semi(A, X) :-
    %$omp target data map(to: A)
    ( X = A ^ elem(0), X > 0 ).
% CHECK: reject-data.m:[[@LINE-2]]:5: error: the goal of a target data directive must be det, but it is semidet
copied_to(!A) :-
    %$omp target data map(to: A)
    !A ^ elem(0) := 1.
% CHECK: reject-data.m:[[@LINE-2]]:5: error: the goal of the target data directive updates A, whose map type does not copy it back; use tofrom or from
unused(A, B, X) :-
    %$omp target data map(to: A) map(to: B)
    X = A ^ elem(0).
% CHECK: reject-data.m:[[@LINE-2]]:5: error: B is not a variable or state variable that the goal of the target data directive uses
twice(A, X) :-
    %$omp target data map(to: A) map(tofrom: A)
    X = A ^ elem(0).
% CHECK: reject-data.m:[[@LINE-2]]:5: error: A is mapped twice
