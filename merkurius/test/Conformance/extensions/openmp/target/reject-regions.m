% DESIGN: Target regions
% A device holds only flat values, so a value that can never be flat, a
% string, a closure, a term of a recursive type such as a list or a tree, or
% anything that holds one, cannot cross into a target region: the error
% names the variable and its type. A region has no way to fail, so its goal
% must be det; and a target teams region with no loop runs its goal in
% every team, so it may not bind a variable used after it.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_regions.
:- interface.
:- import_module array, list.
:- type tree ---> leaf ; node(tree, int, tree).
:- type named ---> named(string, int).
:- pred length(list(float)::in, int::out) is det.
:- pred size(string::in, int::out) is det.
:- pred apply((func(int) = int)::in, int::in, int::out) is det.
:- pred root(tree::in, int::out) is det.
:- pred number(named::in, int::out) is det.
:- pred first(array(string)::array_ui, int::out) is det.
:- pred positive(int::in, int::out) is semidet.
:- pred teams(int::in, int::out) is det.
:- implementation.
:- import_module int, string.

length(Xs, N) :-
    %$omp target
    N = list.length(Xs).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: Xs has type list.list(float), which cannot be mapped to a device
size(S, N) :-
    %$omp target
    N = string.length(S).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: S has type string, which cannot be mapped to a device
apply(F, X, Y) :-
    %$omp target
    Y = F(X).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: F has type (func(int) = int), which cannot be mapped to a device
root(T, R) :-
    %$omp target
    ( T = node(_, R, _) ; T = leaf, R = 0 ).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: T has type reject_regions.tree, which cannot be mapped to a device
number(Named, N) :-
    %$omp target
    Named = named(_, N).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: Named has type reject_regions.named, which cannot be mapped to a device
first(A, N) :-
    %$omp target
    N = string.length(A ^ elem(0)).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: A has type array.array(string), which cannot be mapped to a device
positive(X, Y) :-
    %$omp target
    ( X > 0, Y = X ).
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: the goal of a target directive must be det, but it is semidet
teams(X, Y) :-
    %$omp target teams
    Y = X + 1.
% CHECK: reject-regions.m:[[@LINE-2]]:5: error: a target teams region must not bind Y, which is used after it, since every team would bind it
