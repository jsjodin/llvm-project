% A parallel do loop's body may only read, with from_open, an open value
% that its iterations share (design document, "Open terms", Checking rules;
% docs/INPROGRESS.md, Open types, step 1(c)): the trail is the process's,
% and two iterations binding one variable would race. A get_structure may
% bind it, and so may a call, or a term made from it.
%
% RUN: not merkc -fc1 -fopen-types -fopenmp -fdo-loops -emit-merk %s -o /dev/null 2>&1 \
% RUN:   | FileCheck --implicit-check-not=error: %s

:- module omp_parallel_do_open.
:- interface.
:- import_module list, open, pair.
:- pred bind(open(pair(int, int))::ia, int::in) is det.
:- pred pass(open(pair(int, int))::ia, int::in) is det.
:- pred wrap(open(pair(int, int))::ia, int::in) is det.
:- implementation.
:- import_module int.

bind(X, N) :-
    %$omp parallel do
    ( for(_, 1, N) do X = _ - _ ).
% CHECK: omp-parallel-do-open.m:[[@LINE-1]]:27: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

pass(X, N) :-
    %$omp parallel do
    ( for(_, 1, N) do use(X) ).
% CHECK: omp-parallel-do-open.m:[[@LINE-1]]:23: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

wrap(X, N) :-
    %$omp parallel do
    ( for(_, 1, N) do use_list([X]) ).
% CHECK: omp-parallel-do-open.m:[[@LINE-1]]:32: error: the body of a parallel do loop may only read an open value that its iterations share, with from_open, since another use may bind it, but this uses X

:- pred use(open(pair(int, int))::ia) is det.
use(_).
:- pred use_list(list(open(pair(int, int)))::in(list(any))) is det.
use_list(_).
