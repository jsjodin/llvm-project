% Loops under distribute in a GPU process, with loops under parallel do in
% their bodies (docs/INPROGRESS.md, GPU processes that run loops, and the
% demo's step 3): the process's teams take the distribute loop's
% iterations, a worker team's first wave runs each iteration's body, and a
% parallel do in it runs across the team's threads. A tiler answers each
% number of tiles T, of S cells each, with the checksums of four loops:
% - a team loop in a predicate that the body calls, which fills its tile's
%   cells and sums them, with more cells than a team has threads;
% - a body that reads and updates its own element of an array on both
%   sides of two team loops, one with a minimum, a maximum and a sum of
%   floats, and one with no iterations, whose sum is the value before it;
% - a body that makes an array, fills it in one team loop, reads it, and
%   sums it in another, whose bound is what it read: the first wave's lanes
%   must make one array between them, not one each, which only one lane's
%   words would reach;
% - a loop under distribute parallel do over all the cells, by their index.
% The program runs the kind at the host here, where distribute loops are
% parallel do loops, and at the GPU in process-team-loops-gpu.test.
%
% mmc has no process module to run it with, so the output was computed by
% a Python program written for it, not kept.
%
% The device's compile runs a distribute loop through
% merk_device_process_distribute, its rounds without a barrier, as only a
% team's first wave runs them; and a parallel do in its body through
% merk_device_team_loop, with its words in the team's local memory, from
% merk_device_team_values and merk_device_team_results, and the loops'
% cases in merk_team_loop, apart from merk_process_loop's, so that the
% kernel's calls do not recurse.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential
% RUN: %libomp-env %t.sequential host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t.sequential2
% RUN: %libomp-env %t.sequential2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_team_loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

    % The cells of a tile.
:- func cells = int.
cells = 300.

    % Answers each number of tiles with the checksums of its loops, until 0.
:- process tiler(mailbox(array(int))) receives int on gpu.

tiler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(T), T > 0 then
        S = cells,
        array.init(T * S, 0, Cells0),
        array.init(T, 0, Sums0),
        fill(T, S, Cells0, Cells, Sums0, Sums),
        array.init(T, 1, Updated0),
        update(T, S, Updated0, Updated),
        array.init(T, 0, Made0),
        make(T, S, Made0, Made),
        weigh(T * S, Cells, CellsSum),
        array.init(7, 0, Results0),
        Results = ((((((Results0 ^ elem(0) := CellsSum)
            ^ elem(1) := checksum(T, Sums))
            ^ elem(2) := checksum(T, Updated))
            ^ elem(3) := checksum(T, Made))
            ^ elem(4) := Sums ^ elem(T - 1))
            ^ elem(5) := Updated ^ elem(T - 1))
            ^ elem(6) := Made ^ elem(T - 1),
        send(Reply, Results, !IO),
        tiler(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's cells and their sum.
    %
:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.
% DEVICE-LABEL: merk.proc private @process_team_loops.fill_6_0(
% DEVICE:         merk.foreign_call @merk_device_loop_values() : () -> !llvm.ptr
% DEVICE-NOT:     merk_device_loop_results
% DEVICE:         merk.foreign_call @merk_device_process_distribute(%c0_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}, %{{.*}}) : (i64, i64, i64, i64) -> ()
fill(T, S, !Cells, !Sums) :-
    %$omp distribute
    ( for(J, 0, T - 1) do
        fill_tile(J, S, !Cells, Sum),
        !Sums ^ unsafe_elem(J) := Sum
    ).

    % Tile J's cells, J * 1000 + K for its cell K, and their sum, across the
    % team's threads.
    %
:- pred fill_tile(int::in, int::in, array(int)::array_di,
    array(int)::array_uo, int::out) is det.
% DEVICE-LABEL: merk.proc private @process_team_loops.fill_tile_5_0(
% DEVICE:         merk.foreign_call @merk_device_team_values() : () -> !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_team_results() : () -> !llvm.ptr
% DEVICE:         merk.foreign_call @merk_device_team_loop(%c1_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}, %{{.*}}, %c1_i64{{.*}}) : (i64, i64, i64, i64, i64) -> ()
fill_tile(J, S, !Cells, Sum) :-
    %$omp parallel do reduction((+) : Sum)
    ( for(K, 0, S - 1), fromto(0, P0, P1, Sum) do
        V = J * 1000 + K,
        !Cells ^ unsafe_elem(J * S + K) := V,
        P1 = P0 + V
    ).

    % Each tile's element, read and updated around two team loops.
    %
:- pred update(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
update(T, S, !Updated) :-
    %$omp distribute
    ( for(J, 0, T - 1) do
        !Updated ^ unsafe_elem(J) := !.Updated ^ unsafe_elem(J) * 3 + J,
        %$omp parallel do reduction(min : Lo) reduction(max : Hi) reduction((+) : Half)
        ( for(K, 1, S), fromto(max_int, Lo0, Lo1, Lo),
            fromto(min_int, Hi0, Hi1, Hi), fromto(0.0, H0, H1, Half) do
            V = (K * 7919 + J) mod 1009,
            Lo1 = int.min(Lo0, V),
            Hi1 = int.max(Hi0, V),
            H1 = H0 + float(K) * 0.5
        ),
        %$omp parallel do reduction((+) : None)
        ( for(K, 1, 0), fromto(J, N0, N1, None) do
            N1 = N0 + K
        ),
        !Updated ^ unsafe_elem(J) := !.Updated ^ unsafe_elem(J) * 1000000000 +
            Lo * 1000000 + Hi * 1000 + truncate_to_int(Half) mod 1000 + None
    ).

    % Each tile's sum of an array that its body makes and fills, up to the
    % last element, as the body reads it.
    %
:- pred make(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
make(T, S, !Made) :-
    %$omp distribute
    ( for(J, 0, T - 1) do
        array.init(S, J, Tile0),
        %$omp parallel do
        ( for(K, 0, S - 1), fromto(Tile0, X0, X1, Tile) do
            X1 = X0 ^ unsafe_elem(K) := X0 ^ unsafe_elem(K) + K
        ),
        Last = Tile ^ unsafe_elem(S - 1) - J,
        %$omp parallel do reduction((+) : Total)
        ( for(K, 0, Last), fromto(0, Q0, Q1, Total) do
            Q1 = Q0 + Tile ^ unsafe_elem(K)
        ),
        !Made ^ unsafe_elem(J) := Total
    ).

    % The sum of the cells, each by its index plus one, across the process's
    % teams.
    %
:- pred weigh(int::in, array(int)::array_ui, int::out) is det.
weigh(N, Cells, Sum) :-
    %$omp distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0, P0, P1, Sum) do
        P1 = P0 + Cells ^ unsafe_elem(I) * (I + 1)
    ).

    % The elements, each by its index plus one.
    %
:- func checksum(int, array(int)) = int.
checksum(T, A) = Sum :-
    ( for(J, 0, T - 1), fromto(0, P0, P1, Sum) do
        P1 = P0 + A ^ elem(J) * (J + 1)
    ).

% The device's compile: merk_process_loop has the cases of the loops across
% the process's teams, whose distribute loops' rounds cross no barrier, and
% merk_team_loop those of the loops across a team's threads, whose rounds
% do.
% DEVICE-LABEL: merk.proc @merk_process_loop(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) attributes {merk.process_root
% DEVICE:         merk.switch %arg0 : i64 cases {{\[\[}}0], [2], [3], [4]]
% DEVICE:         merk.call @[[FILL:process_team_loops.fill__body_[0-9_]+]]$rounds(
% DEVICE:       merk.proc private @[[FILL]]$rounds(
% DEVICE-NOT:     merk_device_sync_threads
% DEVICE:         merk.call @[[FILL]]$rounds(
% DEVICE-LABEL: merk.proc @merk_team_loop(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) attributes {merk.process_root
% DEVICE:         merk.switch %arg0 : i64 cases {{\[\[}}1], [5], [6], [7], [8]]
% DEVICE:         merk.call @[[TILE:process_team_loops.fill_tile__body_[0-9_]+]]$rounds(
% DEVICE:       merk.proc private @[[TILE]]$rounds(
% DEVICE:         merk.foreign_call @merk_device_sync_threads()
% DEVICE:         merk.call @[[TILE]]$rounds(

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reply, !IO),
    spawn(Place, tiler(Reply), Tiler, !IO),
    ask(Tiler, Reply, [1, 7, 400], !IO),
    send(Tiler, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(array(int))::in, list(int)::in,
    io::di, io::uo) is det.
ask(_, _, [], !IO).
ask(Tiler, Reply, [T | Ts], !IO) :-
    send(Tiler, T, !IO),
    receive(Reply, any, after(60000), R, !IO),
    ( if R = message(A) then
        io.format("%d tiles: %d %d %d %d, last %d %d %d\n",
            [i(T), i(A ^ elem(0)), i(A ^ elem(1)), i(A ^ elem(2)),
            i(A ^ elem(3)), i(A ^ elem(4)), i(A ^ elem(5)), i(A ^ elem(6))],
            !IO)
    else
        io.write_string("no answer\n", !IO)
    ),
    ask(Tiler, Reply, Ts, !IO).

% CHECK:1 tiles: 8999900 44850 3002006575 44850, last 44850 3002006575 44850
% CHECK-NEXT:7 tiles: 9483704300 34855800 196062202212 1289400, last 1844850 9003007581 46650
% CHECK-NEXT:400 tiles: 1917486278960000 6403556970000 21573991660097200 9996930000, last 119744850 402004008974 164550
