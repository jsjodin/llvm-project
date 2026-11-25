% DESIGN: GPU processes / GPU processes that run loops
% A GPU process runs loops under OpenMP's directives across its kernel's
% teams and threads: teams distribute over tiles, each tile's parallel do
% with sum, min and max reductions; distribute parallel do over a whole
% array, a smoothing step after step, each reading the last step's array
% and writing a new one; a distribute loop with no iterations; and an array
% of two million elements. Run with no argument the kind runs at the host,
% where distribute loops are parallel do loops, which the oracle checks on
% four threads and without -fopenmp; run-gpu-team-loops-gpu.test runs it
% on the GPU and checks the same output.
% MERKC-FLAGS: -fopenmp -fdo-loops -fprocess-declarations
% TRANSLATION: Inputs/gpu_team_loops_translation.erl
%
% REQUIRES: libomp
% RUN: merkc -O0 -fopenmp -fdo-loops -fprocess-declarations %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -fprocess-declarations %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fprocess-declarations %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:tiles(1, 1): message(tiled(0, 0, 0, 0))
% OUTPUT-NEXT:tiles(5, 300): message(tiled(22476995, 0, 9998, 1497183))
% OUTPUT-NEXT:tiles(500, 70): message(tiled(865667486, 0, 10006, 341031))
% OUTPUT-NEXT:tiles(0, 10): message(tiled(0, 9223372036854775807, -9223372036854775808, -1))
% OUTPUT-NEXT:smooth(10, 0): message(smoothed(2310, 25))
% OUTPUT-NEXT:smooth(1000, 20): message(smoothed(550670331, 319210))
% OUTPUT-NEXT:big(2000000): message(summed(999000000))
% OUTPUT-NOT:{{.}}
:- module run_gpu_team_loops.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type job
    --->    tiles(int, int)         % tiles, cells in each
    ;       smooth(int, int)        % cells, steps
    ;       big(int)
    ;       stop.

:- type answer
    --->    tiled(int, int, int, int)   % sums' checksum, lowest, highest, last
    ;       smoothed(int, int)          % checksum, middle cell
    ;       summed(int).

:- process solver(mailbox(answer)) receives job on gpu.

solver(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(tiles(T, S)) then
        array.init(T, 0, Sums0),
        array.init(T, 0, Los0),
        array.init(T, 0, His0),
        tiles(T, S, Sums0, Sums, Los0, Los, His0, His),
        send(Reply, tiled(checksum(T, Sums), lowest(T, Los),
            highest(T, His), ( if T > 0 then Sums ^ elem(T - 1) else -1 )),
            !IO),
        solver(Reply, Self, !IO)
    else if R = message(smooth(N, Steps)) then
        generate(N, Start),
        smooth(Steps, N, Start, End),
        send(Reply, smoothed(checksum(N, End), End ^ elem(N / 2)), !IO),
        solver(Reply, Self, !IO)
    else if R = message(big(N)) then
        array.init(N, 0, A0),
        fill_indices(N, A0, A),
        weigh(N, A, Sum),
        send(Reply, summed(Sum), !IO),
        solver(Reply, Self, !IO)
    else
        true
    ).

    % Each tile's sum, lowest and highest value, its cells across a team's
    % threads.
:- pred tiles(int::in, int::in,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo,
    array(int)::array_di, array(int)::array_uo) is det.

tiles(T, S, !Sums, !Los, !His) :-
    %$omp teams distribute
    ( for(J, 0, T - 1) do
        %$omp parallel do reduction((+) : Sum) reduction(min : Lo) reduction(max : Hi)
        ( for(K, 0, S - 1), fromto(0, P0, P1, Sum),
            fromto(max_int, L0, L1, Lo), fromto(min_int, H0, H1, Hi) do
            V = (J * 7919 + K * 104729) mod 10007,
            P1 = P0 + V,
            L1 = int.min(L0, V),
            H1 = int.max(H0, V)
        ),
        !Sums ^ unsafe_elem(J) := Sum,
        !Los ^ unsafe_elem(J) := Lo,
        !His ^ unsafe_elem(J) := Hi
    ).

:- pred generate(int::in, array(int)::array_uo) is det.

generate(N, A) :-
    array.init(N, 0, A0),
    %$omp teams distribute parallel do
    ( for(I, 0, N - 1), fromto(A0, X0, X1, A) do
        X1 = X0 ^ unsafe_elem(I) := (I * I) mod 1000
    ).

    % Steps smoothing steps: each inner cell becomes the sum of itself and
    % its neighbours, modulo 1000003; the end cells stay.
:- pred smooth(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.

smooth(Steps, N, !A) :-
    ( if Steps = 0 then
        true
    else
        array.copy(!.A, New0),
        step(N, !.A, New0, New),
        !:A = New,
        smooth(Steps - 1, N, !A)
    ).

:- pred step(int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo) is det.

step(N, Old, !New) :-
    %$omp teams distribute parallel do
    ( for(I, 1, N - 2) do
        !New ^ unsafe_elem(I) := (Old ^ unsafe_elem(I - 1) +
            Old ^ unsafe_elem(I) + Old ^ unsafe_elem(I + 1)) mod 1000003
    ).

:- pred fill_indices(int::in, array(int)::array_di, array(int)::array_uo)
    is det.

fill_indices(N, !A) :-
    %$omp teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := I mod 1000
    ).

:- pred weigh(int::in, array(int)::array_ui, int::out) is det.

weigh(N, A, Sum) :-
    %$omp teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0, P0, P1, Sum) do
        P1 = P0 + A ^ unsafe_elem(I)
    ).

    % The elements, each by its index plus one.
:- func checksum(int, array(int)) = int.

checksum(N, A) = Sum :-
    ( for(I, 0, N - 1), fromto(0, P0, P1, Sum) do
        P1 = (P0 + A ^ elem(I) * (I + 1)) mod 1000000007
    ).

:- func lowest(int, array(int)) = int.

lowest(N, A) = Lo :-
    ( for(I, 0, N - 1), fromto(max_int, L0, L1, Lo) do
        L1 = int.min(L0, A ^ elem(I))
    ).

:- func highest(int, array(int)) = int.

highest(N, A) = Hi :-
    ( for(I, 0, N - 1), fromto(min_int, H0, H1, Hi) do
        H1 = int.max(H0, A ^ elem(I))
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reply, !IO),
    spawn(Place, solver(Reply), S, !IO),
    ask(S, Reply, tiles(1, 1), !IO),
    ask(S, Reply, tiles(5, 300), !IO),
    ask(S, Reply, tiles(500, 70), !IO),
    ask(S, Reply, tiles(0, 10), !IO),
    ask(S, Reply, smooth(10, 0), !IO),
    ask(S, Reply, smooth(1000, 20), !IO),
    ask(S, Reply, big(2000000), !IO),
    send(S, stop, !IO).

:- pred ask(mailbox(job)::in, mailbox(answer)::in, job::in,
    io::di, io::uo) is det.

ask(Solver, Reply, Job, !IO) :-
    send(Solver, Job, !IO),
    receive(Reply, any, infinity, R, !IO),
    io.write(Job, !IO),
    io.write_string(": ", !IO),
    io.write_line(R, !IO).
