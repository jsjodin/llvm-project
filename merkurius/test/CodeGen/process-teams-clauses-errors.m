% The clauses of teams that loops across a GPU process's teams do not have
% yet (docs/INPROGRESS.md, GPU processes that run loops): num_teams, which
% would run a loop on fewer of the process's worker teams, and
% thread_limit. Both compiles report them, on the host too, where the loops
% are parallel do loops. Each is in a predicate of its own, as the first
% that is not yet implemented ends a procedure's code.
%
% RUN: not merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fopenmp -fdo-loops -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

:- module process_teams_clauses_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        some_teams(N, A),
        some_threads(N, A, B),
        send(Reply, array.lookup(B, 0), !IO)
    else
        true
    ).

:- pred some_teams(int::in, array(int)::array_uo) is det.

some_teams(N, A) :-
    array.init(N, 0, A0),
    %$omp teams distribute parallel do num_teams(2)
% CHECK: process-teams-clauses-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the num_teams clause of the teams directive
    ( for(I, 0, N - 1), fromto(A0, A1, A2, A) do
        array.set(I, I, A1, A2)
    ).

:- pred some_threads(int::in, array(int)::array_di, array(int)::array_uo)
    is det.

some_threads(N, !A) :-
    %$omp teams distribute parallel do thread_limit(64)
% CHECK: process-teams-clauses-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for the thread_limit clause of the teams directive
    ( for(I, 0, N - 1) do
        !A ^ elem(I) := I
    ).

main(!IO) :-
    io.write_int(1, !IO),
    io.nl(!IO).
