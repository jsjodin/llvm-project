% Errors in loops that GPU processes run across teams (docs/INPROGRESS.md,
% GPU processes that run loops). A loop under distribute parallel do is in
% the code of a process kind declared on gpu, or a predicate of the module
% that it calls, and no target region runs it; the code of a kind that runs
% on a GPU has no other directive, as its kernel is compiled without OpenMP.
% Each compile checks it.
%
% RUN: not merkc -fc1 -fopenmp -fdo-loops -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fprocess-loops -fdo-loops -triple x86_64-unknown-linux-gnu -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OPTION
% OPTION: error: -fprocess-loops is given only with -fprocess-kernels

:- module process_loops_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        array.init(N, 0, A0),
        %$omp parallel do
% CHECK: process-loops-errors.m:[[@LINE-1]]:9: error: a process kind that runs on a GPU runs its loops across its teams under distribute parallel do, and its code has no parallel do directive
        ( for(I, 0, N - 1), fromto(A0, A1, A2, A) do
            array.set(I, I, A1, A2)
        ),
        send(Reply, array.lookup(A, 0), !IO)
    else
        true
    ).

    % Fills an array, in a loop that a target region runs.
:- pred fill(int::in, array(int)::uo) is det.

fill(N, A) :-
    array.init(N, 0, A0),
    %$omp distribute parallel do
% CHECK: process-loops-errors.m:[[@LINE-1]]:5: error: not yet implemented: a distribute parallel do loop in a target region; in a GPU process's code it runs across the process's teams
    ( for(I, 0, N - 1), fromto(A0, A1, A2, A) do
        array.set(I, I, A1, A2)
    ).

main(!IO) :-
    array.init(10, 0, B0),
    %$omp distribute parallel do
% CHECK: process-loops-errors.m:[[@LINE-1]]:5: error: a distribute parallel do loop runs across the teams of a GPU process, so it must be in the code of a process kind declared on gpu, or in a predicate of the module that such a kind calls
    ( for(I, 0, 9), fromto(B0, B1, B2, B) do
        array.set(I, I, B1, B2)
    ),
    %$omp target
    (
        fill(5, C),
        X = array.lookup(C, 0)
    ),
    io.write_int(array.lookup(B, 0) + X, !IO).
