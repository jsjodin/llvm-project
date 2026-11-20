% Flat records that loops in a GPU process take (docs/EXPERIMENTS.md, 2c): a
% loop's body takes a record of integers and floats in a word for each
% field, which the process takes apart and each thread makes again, rather
% than in one word, the record boxed in a cell. Under parallel do in a
% distribute loop's body, the team's first wave made that cell with the
% GPU's malloc in every iteration of the distribute loop, and kept it until
% the process ended. A scaler answers each number of tiles T, of S cells
% each, with the checksums of three loops that take params, whose fields
% come from T:
% - a distribute loop whose body calls a predicate with a team loop, which
%   fills its tile's cells and sums them;
% - a loop under distribute parallel do over all the cells;
% - a distribute loop whose team loop takes an outer, a record with a
%   record in it, which is not flat by this rule and stays one word.
% The program runs the kind at the host here, where distribute loops are
% parallel do loops, and at the GPU in process-loop-records-gpu.test.
%
% mmc has no process module to run it with, so the output was computed by
% a Python program written for it, not kept.
%
% The device's compile takes params apart with merk.deconstruct before the
% loops' words, with no merk_device_alloc, and makes it again in the
% rounds, with merk.construct, before the body's call.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fc1 -fprocess-loops -fdo-loops -fprocess-declarations -fprocess-kernels -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -emit-merk %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_loop_records.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

    % A flat record: a word for each field.
:- type params
    --->    params(
                scale :: int,
                shift :: int,
                weight :: float
            ).

    % A record with a record in it: one word, its cell.
:- type outer
    --->    outer(
                inner :: params,
                extra :: int
            ).

    % The cells of a tile.
:- func cells = int.
cells = 300.

    % Answers each number of tiles with the checksums of its loops, until 0.
:- process scaler(mailbox(array(int))) receives int on gpu.

scaler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(T), T > 0 then
        S = cells,
        P = params(T mod 5 + 2, T, float(T) / 4.0),
        array.init(T * S, 0, Cells0),
        array.init(T, 0, Sums0),
        fill(P, T, S, Cells0, Cells, Sums0, Sums),
        weigh(P, T * S, Cells, Weight),
        array.init(T, 0, Nested0),
        nest(outer(P, T * 3), T, S, Nested0, Nested),
        array.init(4, 0, Results0),
        Results = (((Results0 ^ elem(0) := checksum(T, Sums))
            ^ elem(1) := Weight)
            ^ elem(2) := checksum(T, Nested))
            ^ elem(3) := Sums ^ elem(T - 1),
        send(Reply, Results, !IO),
        scaler(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's cells and their sum.
    %
:- pred fill(params::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo, array(int)::array_di, array(int)::array_uo) is det.
% DEVICE-LABEL: merk.proc private @process_loop_records.fill_{{[0-9_]+}}(
% DEVICE:         merk.deconstruct #merk.functor<"params", 3> %{{.*}} : !merk.term<"process_loop_records.params"> -> i64, i64, f64
% DEVICE:         merk.foreign_call @merk_device_loop_values()
% DEVICE:         merk.foreign_call @merk_device_process_distribute(
fill(P, T, S, !Cells, !Sums) :-
    %$omp teams distribute
    ( for(J, 0, T - 1) do
        fill_tile(P, J, S, !Cells, Sum),
        !Sums ^ unsafe_elem(J) := Sum
    ).

    % Tile J's cells and their sum, across the team's threads: params goes
    % to the team loop's words a field each, not in a cell.
    %
:- pred fill_tile(params::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo, int::out) is det.
% DEVICE-LABEL: merk.proc private @process_loop_records.fill_tile_{{[0-9_]+}}(
% DEVICE-NOT:     merk_device_alloc
% DEVICE:         merk.deconstruct #merk.functor<"params", 3> %{{.*}} : !merk.term<"process_loop_records.params"> -> i64, i64, f64
% DEVICE-NOT:     merk_device_alloc
% DEVICE:         merk.foreign_call @merk_device_team_values()
% DEVICE-NOT:     merk_device_alloc
% DEVICE:         merk.foreign_call @merk_device_team_loop(
fill_tile(P, J, S, !Cells, Sum) :-
    %$omp parallel do reduction((+) : Sum)
    ( for(K, 0, S - 1), fromto(0, Q0, Q1, Sum) do
        V = value(P, J * S + K),
        !Cells ^ unsafe_elem(J * S + K) := V,
        Q1 = Q0 + V
    ).

    % Cell I's value.
    %
:- func value(params, int) = int.
value(P, I) =
    I * P ^ scale + P ^ shift + truncate_to_int(float(I mod 7) * P ^ weight).

    % The cells, each by its index plus the shift, across the process's
    % teams.
    %
:- pred weigh(params::in, int::in, array(int)::array_ui, int::out) is det.
weigh(P, N, Cells, Sum) :-
    %$omp teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0, Q0, Q1, Sum) do
        Q1 = Q0 + Cells ^ unsafe_elem(I) * (I + P ^ shift) * P ^ scale
    ).

    % Each tile's sum over its cells of values from the outer's params and
    % its extra.
    %
:- pred nest(outer::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
nest(O, T, S, !Nested) :-
    %$omp teams distribute
    ( for(J, 0, T - 1) do
        %$omp parallel do reduction((+) : Sum)
        ( for(K, 0, S - 1), fromto(0, Q0, Q1, Sum) do
            Q1 = Q0 + value(O ^ inner, J * S + K) + O ^ extra
        ),
        !Nested ^ unsafe_elem(J) := Sum
    ).

    % The elements, each by its index plus one.
    %
:- func checksum(int, array(int)) = int.
checksum(T, A) = Sum :-
    ( for(J, 0, T - 1), fromto(0, Q0, Q1, Sum) do
        Q1 = Q0 + A ^ elem(J) * (J + 1)
    ).

% The rounds make params again, from its fields, for the body's call.
% DEVICE-LABEL: merk.proc @merk_team_loop(
% DEVICE:       merk.proc private @process_loop_records.fill_tile__body_{{[0-9_]+}}$rounds(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, {{.*}}%{{[a-z0-9]+}}: i64, %{{[a-z0-9]+}}: i64, %{{[a-z0-9]+}}: f64
% DEVICE:         merk.construct #merk.functor<"params", 3>(%{{.*}}, %{{.*}}, %{{.*}} : i64, i64, f64) -> !merk.term<"process_loop_records.params">

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reply, !IO),
    spawn(Place, scaler(Reply), Scaler, !IO),
    ask(Scaler, Reply, [1, 7, 400], !IO),
    send(Scaler, 0, !IO).

:- pred ask(mailbox(int)::in, mailbox(array(int))::in, list(int)::in,
    io::di, io::uo) is det.
ask(_, _, [], !IO).
ask(Scaler, Reply, [T | Ts], !IO) :-
    send(Scaler, T, !IO),
    receive(Reply, any, after(60000), R, !IO),
    ( if R = message(A) then
        io.format("%d tiles: %d %d %d, last %d\n",
            [i(T), i(A ^ elem(0)), i(A ^ elem(1)), i(A ^ elem(2)),
            i(A ^ elem(3))], !IO)
    else
        io.write_string("no answer\n", !IO)
    ),
    ask(Scaler, Reply, Ts, !IO).

% CHECK:1 tiles: 134978 81192858 135878, last 134978
% CHECK-NEXT:7 tiles: 45442848 49708853000 45619248, last 2342962
% CHECK-NEXT:400 tiles: 3864011979600 2325638243840800 3892883979600, last 72119400
