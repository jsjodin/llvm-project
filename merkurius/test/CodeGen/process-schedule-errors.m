% The schedules that loops across a GPU process's teams do not have yet
% (docs/INPROGRESS.md, GPU processes that run loops): schedule(dynamic)
% with a chunk size that is a variable, which the device's compile needs as
% a constant, and schedule(static) with a chunk size. The host's compile,
% where they are parallel do loops, gives omp.wsloop both.
%
% RUN: not merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fc1 -fopenmp -fdo-loops -fprocess-declarations -emit-merk -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=HOST

:- module process_schedule_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- process worker(mailbox(int)) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        dynamic(N, A),
        static(N, A, B),
        send(Reply, array.lookup(B, 0), !IO)
    else
        true
    ).

:- pred dynamic(int::in, array(int)::array_uo) is det.

dynamic(N, A) :-
    C = N / 4 + 1,
    array.init(N, 0, A0),
    %$omp distribute parallel do schedule(dynamic, C)
% CHECK: process-schedule-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for a chunk size that is a variable in a loop across a GPU process's teams
% HOST:    omp.wsloop schedule(dynamic = %{{.*}} : i64) {
    ( for(I, 0, N - 1), fromto(A0, A1, A2, A) do
        array.set(I, I, A1, A2)
    ).

:- pred static(int::in, array(int)::array_di, array(int)::array_uo) is det.

static(N, A, B) :-
    %$omp distribute parallel do schedule(static, 4)
% CHECK: process-schedule-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for schedule(static) with a chunk size in a loop across a GPU process's teams
% HOST:    omp.wsloop schedule(static = %{{.*}} : i64) {
    ( for(I, 0, N - 1), fromto(A, B1, B2, B) do
        array.set(I, I + 1, B1, B2)
    ).

main(!IO) :-
    new_mailbox(Me, !IO),
    spawn(gpu(0), worker(Me), W, !IO),
    send(W, 8, !IO),
    receive(Me, any, infinity, _, !IO).
