% Loops under schedule(dynamic) in a GPU process (docs/INPROGRESS.md, GPU
% processes that run loops): each thread claims its iterations, a chunk at a
% time, from a word after the loop's words, which the process sets to 0
% before each run, and goes on as soon as its own end, with no barrier after
% each round. A scheduler answers each number N with what five loops made:
% - under distribute parallel do schedule(dynamic), iterations of uneven
%   work, I mod 97 steps each, with three reductions: the work's sum, the
%   counters' sum and a count, so an iteration run twice or not at all
%   shows;
% - under schedule(dynamic, 3), each element of an array set to the square
%   of its index mod 1009, in chunks that need not divide N;
% - under parallel do schedule(dynamic, 2) in the body of a distribute loop,
%   across a team's threads, a tile of cells filled and summed;
% - under schedule(static), the static schedule's rounds, as with none.
% The same loops run for each number, so each run must start its claims
% from 0. The program runs the kind at the host here, where the loops are
% parallel do loops whose omp.wsloop has the schedule, or sequential without
% -fopenmp, and at the GPU in process-schedule-gpu.test.
%
% mmc has no process module to run it with, so the output was computed by
% a Python program written for it, not kept.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc -fc1 -fopenmp -fdo-loops -fprocess-declarations -emit-merk %s -o - | FileCheck %s --check-prefix=HOST
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env OMP_NUM_THREADS=4 %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t.sequential
% RUN: %libomp-env %t.sequential host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t.sequential2
% RUN: %libomp-env %t.sequential2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_schedule.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process.

    % The cells of a tile.
:- func cells = int.
cells = 50.

    % Answers each number with what its loops made, until 0.
:- process scheduler(mailbox(array(int))) receives int on gpu.

scheduler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N), N > 0 then
        uneven(N, Work, Counters, Count),
        array.init(N, -1, Squares0),
        squares(N, Squares0, Squares),
        T = N / 10 + 1,
        array.init(T * cells, 0, Cells0),
        array.init(T, 0, Sums0),
        tiles(T, Cells0, Cells, Sums0, Sums),
        static_sum(N, Static),
        array.init(7, 0, Results0),
        Results = ((((((Results0 ^ elem(0) := Work)
            ^ elem(1) := Counters)
            ^ elem(2) := Count)
            ^ elem(3) := checksum(N, Squares))
            ^ elem(4) := checksum(T * cells, Cells))
            ^ elem(5) := checksum(T, Sums))
            ^ elem(6) := Static,
        send(Reply, Results, !IO),
        scheduler(Reply, Self, !IO)
    else
        true
    ).

    % Iterations of uneven work, claimed one at a time.
    %
:- pred uneven(int::in, int::out, int::out, int::out) is det.
% DEVICE-LABEL: merk.proc private @process_schedule.uneven_4_0(
% DEVICE:         %[[VALUES:.*]] = merk.foreign_call @merk_device_loop_values() : () -> !llvm.ptr
% DEVICE:         %[[CLAIMS:.*]] = llvm.getelementptr %[[VALUES]][0] : (!llvm.ptr) -> !llvm.ptr, i64
% DEVICE:         llvm.store %c0_i64{{.*}}, %[[CLAIMS]] : i64, !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_process_loop(
% HOST-LABEL:   merk.proc private @process_schedule.uneven_4_0(
% HOST:           omp.wsloop schedule(dynamic) reduction(
uneven(N, Work, Counters, Count) :-
    %$omp teams distribute parallel do schedule(dynamic) reduction((+) : Work)
    %$omp& reduction((+) : Counters) reduction((+) : Count)
    ( for(I, 0, N - 1), fromto(0, W0, W1, Work),
        fromto(0, C0, C1, Counters), fromto(0, K0, K1, Count) do
        ( for(K, 0, I mod 97), fromto(W0, V0, V1, W1) do
            V1 = V0 + (I * K) mod 13
        ),
        C1 = C0 + I,
        K1 = K0 + 1
    ).

    % Each element the square of its index mod 1009, three at a time.
    %
:- pred squares(int::in, array(int)::array_di, array(int)::array_uo) is det.
% HOST-LABEL:   merk.proc private @process_schedule.squares_3_0(
% HOST:           %[[CHUNK:.*]] = arith.constant 3 : i64
% HOST:           omp.wsloop schedule(dynamic = %[[CHUNK]] : i64) {
squares(N, !Squares) :-
    %$omp teams distribute parallel do schedule(dynamic, 3)
    ( for(I, 0, N - 1), fromto(!.Squares, S0, S1, !:Squares) do
        S1 = S0 ^ unsafe_elem(I) := (I mod 1009) * (I mod 1009)
    ).

    % Each tile's cells, J * 100 + K for its cell K, and their sum, its
    % cells claimed two at a time across the team's threads.
    %
:- pred tiles(int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.
tiles(T, !Cells, !Sums) :-
    %$omp teams distribute
    ( for(J, 0, T - 1) do
        tile(J, !Cells, Sum),
        !Sums ^ unsafe_elem(J) := Sum
    ).

:- pred tile(int::in, array(int)::array_di, array(int)::array_uo, int::out)
    is det.
% DEVICE-LABEL: merk.proc private @process_schedule.tile_4_0(
% DEVICE:         %[[TEAM:.*]] = merk.foreign_call @merk_device_team_values() : () -> !llvm.ptr
% DEVICE:         llvm.store %c0_i64{{.*}} : i64, !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_team_loop(
tile(J, !Cells, Sum) :-
    S = cells,
    %$omp parallel do schedule(dynamic, 2) reduction((+) : Sum)
    ( for(K, 0, S - 1), fromto(0, P0, P1, Sum) do
        V = J * 100 + K,
        !Cells ^ unsafe_elem(J * S + K) := V,
        P1 = P0 + V
    ).

    % The counters' sum under the static schedule.
    %
:- pred static_sum(int::in, int::out) is det.
% HOST-LABEL:   merk.proc private @process_schedule.static_sum_2_0(
% HOST:           omp.wsloop schedule(static) reduction(
static_sum(N, Sum) :-
    %$omp teams distribute parallel do schedule(static) reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0, P0, P1, Sum) do
        P1 = P0 + I * 3
    ).

    % The elements, each by its index plus one.
    %
:- func checksum(int, array(int)) = int.
checksum(N, A) = Sum :-
    ( for(J, 0, N - 1), fromto(0, P0, P1, Sum) do
        P1 = P0 + A ^ elem(J) * (J + 1)
    ).

% The device's compile: a dynamic loop's case calls its claims, with the
% address of its claims word, the same in every lane, whose procedure claims
% a chunk with one atomic add, on a global pointer for a loop across the
% process's teams and on the team's local memory for a team loop, and runs
% it through the rounds of a chunk, which cross no barrier; a static loop's
% rounds cross one after each round.
% DEVICE-LABEL: merk.proc @merk_process_loop(
% DEVICE:         merk.call @[[UNEVEN:process_schedule.uneven__body_[0-9_]+]]$claims(
% DEVICE:         merk.call @[[SQUARES:process_schedule.squares__body_[0-9_]+]]$claims(
% DEVICE:         merk.call @[[STATIC:process_schedule.static_sum__body_[0-9_]+]]$rounds(
% DEVICE:       merk.proc private @[[UNEVEN]]$chunk(
% DEVICE-NOT:     merk_device_sync_threads
% DEVICE:         merk.call @[[UNEVEN]]$chunk(
% DEVICE:       merk.proc private @[[UNEVEN]]$claims(
% DEVICE:         %[[GENERIC:.*]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
% DEVICE:         %[[GLOBAL:.*]] = llvm.addrspacecast %[[GENERIC]] : !llvm.ptr to !llvm.ptr<1>
% DEVICE:         llvm.atomicrmw add %[[GLOBAL]], %c1_i64{{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
% DEVICE:         merk.call @[[UNEVEN]]$chunk(
% DEVICE:         merk.call @[[UNEVEN]]$claims(
% DEVICE:       merk.proc private @[[SQUARES]]$claims(
% DEVICE:         llvm.atomicrmw add %{{.*}}, %c3_i64{{.*}} syncscope("agent") monotonic : !llvm.ptr<1>, i64
% DEVICE:       merk.proc private @[[STATIC]]$rounds(
% DEVICE:         merk.foreign_call @merk_device_sync_threads()
% DEVICE-LABEL: merk.proc @merk_team_loop(
% DEVICE:         merk.call @[[TILE:process_schedule.tile__body_[0-9_]+]]$claims(
% DEVICE:       merk.proc private @[[TILE]]$claims(
% DEVICE:         llvm.atomicrmw add %{{.*}}, %c2_i64{{.*}} syncscope("workgroup") monotonic : !llvm.ptr<3>, i64

% In LLVM IR, the claims' address is read from the wave's first lane.
% LLVM-LABEL: define {{.*}} @merk_process_loop(
% LLVM:         call i64 @llvm.amdgcn.readfirstlane.i64(
% LLVM:         atomicrmw add ptr addrspace(1) %{{.*}}, i64 1 syncscope("agent") monotonic

% CHECK:1: 0 0 1 0 41650 1225 0
% CHECK-NEXT:2: 1 1 2 2 41650 1225 3
% CHECK-NEXT:10: 284 45 10 2310 522050 13675 135
% CHECK-NEXT:1000: 262895 499500 1000 249833083500 85544894150 1723309975 1498500
% CHECK-NEXT:100000: 26849508 4999950000 100000 1699190521429500 83355272610291650 1667227968376225 14999850000
% CHECK-NEXT:1000: 262895 499500 1000 249833083500 85544894150 1723309975 1498500

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Me, !IO),
    spawn(Place, scheduler(Me), Scheduler, !IO),
    list.foldl(ask(Scheduler, Me), [1, 2, 10, 1000, 100000, 1000], !IO),
    send(Scheduler, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(array(int))::in, int::in,
    io::di, io::uo) is det.

ask(Scheduler, Me, N, !IO) :-
    send(Scheduler, N, !IO),
    receive(Me, any, after(60000), R, !IO),
    io.write_int(N, !IO),
    io.write_string(":", !IO),
    ( if R = message(Results) then
        array.foldl(print_result, Results, !IO)
    else
        io.write_string(" ", !IO),
        io.print(R, !IO)
    ),
    io.nl(!IO).

:- pred print_result(int::in, io::di, io::uo) is det.

print_result(V, !IO) :-
    io.write_string(" ", !IO),
    io.write_int(V, !IO).
