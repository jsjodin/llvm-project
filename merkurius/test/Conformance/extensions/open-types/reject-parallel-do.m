% DESIGN: Open terms (Prolog-style variables) / Checking rules
% A parallel do body may not bind an open value that its iterations share,
% as the trail belongs to the process and two threads binding a variable
% would race: it may use one only as from_open's argument. An open term
% that an iteration makes is its own, and a loop without the directive is
% sequential.
% ORACLE: none
%
% RUN: not merkc -fc1 -fopen-types -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error: --implicit-check-not=warning:
:- module reject_parallel_do.
:- interface.
:- import_module list, open.
:- pred bind(open(list(int))::ia, int::in) is det.
:- pred unify(open(list(int))::ia, open(list(int))::ia, int::in) is det.
:- pred call_it(open(list(int))::ia, int::in) is det.
:- pred read_only(open(list(int))::ia, int::in, int::out) is det.
:- pred own_term(int::in, int::out) is det.
:- pred sequential(open(list(int))::ia, int::in) is det.
:- implementation.
:- import_module int.

bind(X, N) :-
    %$omp parallel do
    ( for(_, 1, N) do promise_pure ( if X = [_] then true else true ) ).
% CHECK-DAG: reject-parallel-do.m:[[@LINE-1]]:45: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

unify(X, Y, N) :-
    %$omp parallel do
    ( for(_, 1, N) do promise_pure ( if X = Y then true else true ) ).
% CHECK-DAG: reject-parallel-do.m:[[@LINE-1]]:41: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

call_it(X, N) :-
    %$omp parallel do
    ( for(_, 1, N) do use(X) ).
% CHECK-DAG: reject-parallel-do.m:[[@LINE-1]]:23: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

:- pred use(open(list(int))::ia) is det.
use(_).

    % Not an error: from_open only reads the term.
read_only(X, N, S) :-
    %$omp parallel do reduction((+) : S)
    ( for(_, 1, N), fromto(0, S0, S1, S) do
        promise_pure (
            if impure from_open(X, L) then S1 = S0 + list.length(L)
            else S1 = S0
        )
    ).

    % Not an error: each iteration binds an open term of its own.
own_term(N, S) :-
    %$omp parallel do reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do
        promise_pure (
            L = [A] : open(list(int)),
            ( if A = to_open(I), impure from_open(L, [V]) then
                S1 = S0 + V
            else
                S1 = S0
            )
        )
    ).

    % Not an error: without the directive the loop is sequential.
sequential(X, N) :-
    ( for(_, 1, N) do promise_pure ( if X = [_] then true else true ) ).
