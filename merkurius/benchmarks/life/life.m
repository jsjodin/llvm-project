% Conway's Game of Life on one GPU, dense: the reference in Mercury for the
% demo of a stencil whose active region moves (docs/INPROGRESS.md), beside
% life.c, the same program in C, to show what Mercury's code costs before
% any process is involved. See README.md.
%
% The grid is LIFE_SIZE cells square, with a border of dead cells around it,
% in one array of (LIFE_SIZE + 2)^2 ints by rows. A step computes every cell
% from its eight neighbours into the other array, and the two swap. The
% arrays stay on the device for all LIFE_STEPS steps.

:- module life.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

main(!IO) :-
    setting("LIFE_SIZE", 4096, N, !IO),
    setting("LIFE_STEPS", 100, Steps, !IO),
    ( if N < 1 then
        usage(!IO)
    else if Steps < 2 then
        usage(!IO)
    else
        header(N, Steps, !IO),
        W = N + 2,
        init(N, array.init(W * W, 0), A),
        array.init(W * W, 0, B),
        array.init(Steps, 0.0, T0),
        %$omp target data map(tofrom: A, B)
        run(0, Steps, N, A, B, Final, T0, T, !IO),
        count(N, Final, Population, Checksum),
        results(N, Steps, T, Population, Checksum, !IO)
    ).

:- pred usage(io::di, io::uo) is det.
usage(!IO) :-
    io.write_string("LIFE_SIZE must be at least 1 and LIFE_STEPS at least 2.\n",
        !IO),
    exit_failure(!IO).

    % Whether the cell of index K, counting the grid's cells by rows from 0,
    % starts alive: about 30% do, by a multiplicative hash of K.
    %
:- func alive(int) = int.
alive(K) = V :-
    H = (K * 2654435761) /\ 4294967295,
    ( if unchecked_rem(H >> 16, 100) < 30 then V = 1 else V = 0 ).

:- pred init(int::in, array(int)::array_di, array(int)::array_uo) is det.
init(N, !A) :-
    W = N + 2,
    %$omp target teams distribute parallel do
    ( for(K, 0, N * N - 1) do
        C = (unchecked_quotient(K, N) + 1) * W + unchecked_rem(K, N) + 1,
        !A ^ unsafe_elem(C) := alive(K)
    ).

    % One step: B's cells from A's.
    %
:- pred step(int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo) is det.
step(N, A, !B) :-
    W = N + 2,
    %$omp target teams distribute parallel do
    ( for(K, 0, N * N - 1) do
        C = (unchecked_quotient(K, N) + 1) * W + unchecked_rem(K, N) + 1,
        Count =
            A ^ unsafe_elem(C - W - 1) + A ^ unsafe_elem(C - W) +
            A ^ unsafe_elem(C - W + 1) + A ^ unsafe_elem(C - 1) +
            A ^ unsafe_elem(C + 1) + A ^ unsafe_elem(C + W - 1) +
            A ^ unsafe_elem(C + W) + A ^ unsafe_elem(C + W + 1),
        ( if Count = 3 then
            V = 1
        else if Count = 2 then
            V = A ^ unsafe_elem(C)
        else
            V = 0
        ),
        !B ^ unsafe_elem(C) := V
    ).

    % Steps from S to Steps, from Cur into Next, which swap after each,
    % timing each step into T. Final is the grid after the last.
    %
:- pred run(int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo, io::di, io::uo) is det.
run(S, Steps, N, Cur, Next0, Final, !T, !IO) :-
    ( if S >= Steps then
        Final = Cur
    else
        now(T0, !IO),
        step(N, Cur, Next0, Next),
        now(T1, !IO),
        !T ^ elem(S) := T1 - T0,
        run(S + 1, Steps, N, Next, Cur, Final, !T, !IO)
    ).

    % The live cells, and the sum of their indices, counting the grid's
    % cells by rows from 0, which says where they are.
    %
:- pred count(int::in, array(int)::in, int::out, int::out) is det.
count(N, A, Population, Checksum) :-
    W = N + 2,
    ( for(K, 0, N * N - 1),
        fromto(0, P0, P1, Population), fromto(0, S0, S1, Checksum) do
        C = (K / N + 1) * W + K mod N + 1,
        V = A ^ elem(C),
        P1 = P0 + V,
        S1 = S0 + V * K
    ).

    % The time of each step but the first, which includes starting the
    % kernels, as min, max and average, and cells updated a second from
    % the fastest.
    %
:- pred results(int::in, int::in, array(float)::in, int::in, int::in,
    io::di, io::uo) is det.
results(N, Steps, T, Population, Checksum, !IO) :-
    ( for(S, 1, Steps - 1),
        fromto(T ^ elem(1), Min0, Min1, Min),
        fromto(T ^ elem(1), Max0, Max1, Max),
        fromto(0.0, Sum0, Sum1, Sum) do
        Min1 = float.min(Min0, T ^ elem(S)),
        Max1 = float.max(Max0, T ^ elem(S)),
        Sum1 = Sum0 + T ^ elem(S)
    ),
    Average = Sum / float(Steps - 1),
    Rate = float(N) * float(N) / Min * 1.0e-9,
    write_results(Population, Checksum, Rate, Min, Max, Average, !IO).

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

:- pred header(int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(N::in, Steps::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Game of Life, Mercury\\n\");
    printf(\"Grid: %ld x %ld cells, %ld bytes a cell\\n\", (long) N, (long) N,
        (long) sizeof(MR_Integer));
    printf(\"Steps: %ld\\n\", (long) Steps);
").

:- pred write_results(int::in, int::in, float::in, float::in, float::in,
    float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Population::in, Checksum::in, Rate::in, Min::in, Max::in,
        Average::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Population: %ld\\n\", (long) Population);
    printf(\"Checksum: %ld\\n\", (long) Checksum);
    printf(\"%-14s%-12s%-12s%-12s\\n\", \"Gcells/sec\", \"Min (sec)\", \"Max\",
        \"Average\");
    printf(\"%-14.3f%-12.6f%-12.6f%-12.6f\\n\", Rate, Min, Max, Average);
").
