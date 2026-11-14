% Foreign code in a GPU process whose locals have their addresses taken: a
% struct that clock_gettime fills, in the process's own code, and a double
% whose bits memcpy copies, in a loop under parallel do in the body of a
% distribute loop. A GPU process's image has no OpenMP device runtime, so
% its foreign code is compiled in clang's CUDA mode, where such a local
% stays on its thread's stack; in the default mode clang put it in memory
% from __kmpc_alloc_shared, which the image's link did not find.
%
% A worker answers each number of tiles T with the sum of the bits of the
% floats J * 64 + K over tiles J and their cells K, by tiles, and whether
% the clock went forward. The program runs it at the host here, and at the
% GPU in process-foreign-locals-gpu.test.
%
% mmc has no process module to run it with, so the output was computed by
% a Python program written for it, not kept.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -O2 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE --implicit-check-not=__kmpc_
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t0
% RUN: %libomp-env %t0 host | FileCheck %s --match-full-lines --strict-whitespace

% The locals are on the thread's stack, in the private address space.
% DEVICE: %ts.{{.*}} = alloca %struct.timespec, align 8, addrspace(5)
% DEVICE: call i32 @clock_gettime(
% DEVICE: %B.{{.*}} = alloca i64, align 8, addrspace(5)

:- module process_foreign_locals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Reply, !IO),
    spawn(Place, worker(Reply), Worker, !IO),
    list.foldl(ask(Worker, Reply), [1, 3, 40], !IO),
    send(Worker, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(array(int))::in, int::in,
    io::di, io::uo) is det.
ask(Worker, Reply, T, !IO) :-
    send(Worker, T, !IO),
    receive(Reply, any, after(60000), R, !IO),
    ( if R = message(A) then
        % The sum's two halves, as %x prints a negative int with a sign.
        io.format("%d tiles: %08x%08x, clock %s\n",
            [i(T), i((A ^ elem(0) >> 32) /\ 0xffffffff),
            i(A ^ elem(0) /\ 0xffffffff),
            s(if A ^ elem(1) = 1 then "forward" else "backward")], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).
% CHECK:1 tiles: cdf8000000000000, clock forward
% CHECK-NEXT:3 tiles: fcec000000000000, clock forward
% CHECK-NEXT:40 tiles: 43ce000000000000, clock forward

    % Answers each number of tiles until 0.
:- process worker(mailbox(array(int))) receives int on gpu.

worker(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(T), T > 0 then
        now(T0, !IO),
        array.init(T, 0, Sums0),
        sums(T, Sums0, Sums),
        now(T1, !IO),
        ( for(J, 0, T - 1), fromto(0, S0, S1, Total) do
            S1 = S0 + Sums ^ elem(J)
        ),
        array.init(2, 0, Results0),
        Results = (Results0 ^ elem(0) := Total)
            ^ elem(1) := ( if T1 >= T0 then 1 else 0 ),
        send(Reply, Results, !IO),
        worker(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's sum, a team for each.
    %
:- pred sums(int::in, array(int)::array_di, array(int)::array_uo) is det.
sums(T, !Sums) :-
    %$omp distribute
    ( for(J, 0, T - 1) do
        tile_sum(J, Sum),
        !Sums ^ unsafe_elem(J) := Sum
    ).

    % The sum of the bits of tile J's floats, across the team's threads.
    %
:- pred tile_sum(int::in, int::out) is det.
tile_sum(J, Sum) :-
    %$omp parallel do reduction((+) : Sum)
    ( for(K, 0, 63), fromto(0, S0, S1, Sum) do
        S1 = S0 + bits(float(J * 64 + K))
    ).

:- pragma foreign_decl("C", "#include <string.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- func bits(float) = int.
:- pragma foreign_proc("C",
    bits(F::in) = (B::out),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    double y = F;
    memcpy(&B, &y, sizeof y);
").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").
