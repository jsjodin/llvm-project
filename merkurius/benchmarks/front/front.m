% A reacting front on one GPU, dense: the reference in Mercury for the demo
% of a stencil whose active region moves (docs/INPROGRESS.md), beside
% front.c, the same program in C. Heat diffuses between cells, and where a
% cell is hot enough, its fuel burns, in a stiff system of two reactions,
% many substeps a step; so the work is heavy where the front is, and light
% elsewhere. See README.md.
%
% The grid is FRONT_SIZE cells square, with a cold border around it, in
% arrays of (FRONT_SIZE + 2)^2 floats by rows: the temperature, twice, as a
% step computes it from its neighbours' into the other array, and the fuel
% and the intermediate, which a step updates in place. The arrays stay on
% the device for all FRONT_STEPS steps.

:- module front.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, string.

    % The model (README.md): diffusion of heat, a loss of heat to the
    % surroundings, and two reactions, fuel to an intermediate and the
    % intermediate to product, each giving heat.
    %
:- func alpha = float.
alpha = 0.2.
:- func loss = float.
loss = 0.01.
:- func ignite1 = float.
ignite1 = 1.5.
:- func ignite2 = float.
ignite2 = 2.0.
:- func rate1 = float.
rate1 = 0.5.
:- func rate2 = float.
rate2 = 2.0.
:- func heat1 = float.
heat1 = 1.0.
:- func heat2 = float.
heat2 = 3.0.
:- func gone = float.
gone = 1.0e-9.

main(!IO) :-
    setting("FRONT_SIZE", 4096, N, !IO),
    setting("FRONT_STEPS", 100, Steps, !IO),
    setting("FRONT_SUBSTEPS", 100, Substeps, !IO),
    ( if N < 1 then
        usage(!IO)
    else if Steps < 2 then
        usage(!IO)
    else if Substeps < 1 then
        usage(!IO)
    else
        header(N, Steps, Substeps, !IO),
        W = N + 2,
        init(N, array.init(W * W, 0.0), T, array.init(W * W, 0.0), U,
            array.init(W * W, 0.0), Y0, array.init(W * W, 0.0), Z0),
        array.init(Steps, 0.0, Times0),
        %$omp target data map(tofrom: T, U, Y0, Z0)
        run(0, Steps, N, Substeps, T, U, Final, Y0, Y, Z0, _, Times0, Times,
            !IO),
        count(N, Final, Y, Burnt, Heat, Fuel),
        results(N, Steps, Times, Burnt, Heat, Fuel, !IO)
    ).

:- pred usage(io::di, io::uo) is det.
usage(!IO) :-
    io.write_string("FRONT_SIZE and FRONT_SUBSTEPS must be at least 1 " ++
        "and FRONT_STEPS at least 2.\n", !IO),
    exit_failure(!IO).

    % Cold and full of fuel, but for a hot disc of radius 5 at a quarter of
    % the way along each side; the border is cold, and stays so.
    %
:- pred init(int::in, array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo) is det.
init(N, !T, !U, !Y, !Z) :-
    W = N + 2,
    %$omp target teams distribute parallel do
    ( for(K, 0, W * W - 1) do
        R = unchecked_quotient(K, W) - 1,
        C = unchecked_rem(K, W) - 1,
        DR = float(R) - float(N) / 4.0,
        DC = float(C) - float(N) / 4.0,
        ( if R >= 0, R < N, C >= 0, C < N then
            ( if DR * DR + DC * DC < 25.0 then
                !T ^ unsafe_elem(K) := 5.0
            else
                !T ^ unsafe_elem(K) := 1.0
            ),
            !Y ^ unsafe_elem(K) := 1.0
        else
            !T ^ unsafe_elem(K) := 1.0,
            !Y ^ unsafe_elem(K) := 0.0
        ),
        !U ^ unsafe_elem(K) := 1.0,
        !Z ^ unsafe_elem(K) := 0.0
    ).

    % One step: U's temperatures from T's, and each cell's fuel and
    % intermediate in place, in Substeps substeps where the cell burns.
    %
:- pred step(int::in, int::in, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo) is det.
step(N, Substeps, T, !U, !Y, !Z) :-
    W = N + 2,
    H = 1.0 / float(Substeps),
    %$omp target teams distribute parallel do
    ( for(K, 0, N * N - 1) do
        C = (unchecked_quotient(K, N) + 1) * W + unchecked_rem(K, N) + 1,
        Heat0 = T ^ unsafe_elem(C) + alpha * (T ^ unsafe_elem(C - W) +
            T ^ unsafe_elem(C + W) + T ^ unsafe_elem(C - 1) +
            T ^ unsafe_elem(C + 1) - 4.0 * T ^ unsafe_elem(C)),
        Heat1 = Heat0 - loss * (Heat0 - 1.0),
        Fuel0 = !.Y ^ unsafe_elem(C),
        Middle0 = !.Z ^ unsafe_elem(C),
        ( if Heat1 > ignite1, Fuel0 + Middle0 > 0.0 then
            burn(Substeps, H, Heat1, Heat, Fuel0, Fuel1, Middle0, Middle1),
            ( if Fuel1 < gone then Fuel = 0.0 else Fuel = Fuel1 ),
            ( if Middle1 < gone then Middle = 0.0 else Middle = Middle1 )
        else
            Heat = Heat1,
            Fuel = Fuel0,
            Middle = Middle0
        ),
        !U ^ unsafe_elem(C) := Heat,
        !Y ^ unsafe_elem(C) := Fuel,
        !Z ^ unsafe_elem(C) := Middle
    ).

    % A burning cell's substeps, each a linearly implicit Euler step of
    % each reaction in turn, which stays stable however fast they are.
    %
:- pred burn(int::in, float::in, float::in, float::out, float::in,
    float::out, float::in, float::out) is det.
burn(Substeps, H, !Heat, !Fuel, !Middle) :-
    ( for(_, 0, Substeps - 1),
        fromto(!.Heat, Heat0, Heat2, !:Heat),
        fromto(!.Fuel, Fuel0, Fuel1, !:Fuel),
        fromto(!.Middle, Middle0, Middle2, !:Middle) do
        ( if Heat0 > ignite1 then
            K1 = rate1 * (Heat0 - ignite1) * (Heat0 - ignite1)
        else
            K1 = 0.0
        ),
        Fuel1 = Fuel0 / (1.0 + H * K1),
        Burnt1 = Fuel0 - Fuel1,
        Middle1 = Middle0 + Burnt1,
        Heat1 = Heat0 + heat1 * Burnt1,
        ( if Heat1 > ignite2 then
            K2 = rate2 * (Heat1 - ignite2)
        else
            K2 = 0.0
        ),
        Middle2 = Middle1 / (1.0 + H * K2),
        Burnt2 = Middle1 - Middle2,
        Heat2 = Heat1 + heat2 * Burnt2
    ).

    % Steps from S to Steps, from Cur into Next, which swap after each,
    % timing each step into Times. Final is the temperatures after the last.
    %
:- pred run(int::in, int::in, int::in, int::in,
    array(float)::array_di, array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo, io::di, io::uo) is det.
run(S, Steps, N, Substeps, Cur, Next0, Final, !Y, !Z, !Times, !IO) :-
    ( if S >= Steps then
        Final = Cur
    else
        now(T0, !IO),
        step(N, Substeps, Cur, Next0, Next, !Y, !Z),
        now(T1, !IO),
        !Times ^ elem(S) := T1 - T0,
        run(S + 1, Steps, N, Substeps, Next, Cur, Final, !Y, !Z, !Times, !IO)
    ).

    % The burnt cells, with less than half their fuel left, and the sums of
    % the temperatures and the fuel, by rows in order, so that every build
    % adds the same way.
    %
:- pred count(int::in, array(float)::in, array(float)::in, int::out,
    float::out, float::out) is det.
count(N, T, Y, Burnt, Heat, Fuel) :-
    W = N + 2,
    ( for(K, 0, N * N - 1),
        fromto(0, B0, B1, Burnt),
        fromto(0.0, H0, H1, Heat),
        fromto(0.0, F0, F1, Fuel) do
        C = (K / N + 1) * W + K mod N + 1,
        ( if Y ^ elem(C) < 0.5 then B1 = B0 + 1 else B1 = B0 ),
        H1 = H0 + T ^ elem(C),
        F1 = F0 + Y ^ elem(C)
    ).

    % The time of each step but the first, which includes starting the
    % kernels, as min, max and average, and cells updated a second from
    % the fastest.
    %
:- pred results(int::in, int::in, array(float)::in, int::in, float::in,
    float::in, io::di, io::uo) is det.
results(N, Steps, Times, Burnt, Heat, Fuel, !IO) :-
    ( for(S, 1, Steps - 1),
        fromto(Times ^ elem(1), Min0, Min1, Min),
        fromto(Times ^ elem(1), Max0, Max1, Max),
        fromto(0.0, Sum0, Sum1, Sum) do
        Min1 = float.min(Min0, Times ^ elem(S)),
        Max1 = float.max(Max0, Times ^ elem(S)),
        Sum1 = Sum0 + Times ^ elem(S)
    ),
    Average = Sum / float(Steps - 1),
    Rate = float(N) * float(N) / Min * 1.0e-9,
    write_results(Burnt, Heat, Fuel, Rate, Min, Max, Average, !IO).

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

:- pred header(int::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(N::in, Steps::in, Substeps::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Reacting front, Mercury\\n\");
    printf(\"Grid: %ld x %ld cells\\n\", (long) N, (long) N);
    printf(\"Steps: %ld, of %ld substeps where a cell burns\\n\",
        (long) Steps, (long) Substeps);
").

:- pred write_results(int::in, float::in, float::in, float::in, float::in,
    float::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Burnt::in, Heat::in, Fuel::in, Rate::in, Min::in, Max::in,
        Average::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Burnt cells: %ld\\n\", (long) Burnt);
    printf(\"Heat: %.15e\\n\", Heat);
    printf(\"Fuel: %.15e\\n\", Fuel);
    printf(\"%-14s%-12s%-12s%-12s\\n\", \"Gcells/sec\", \"Min (sec)\", \"Max\",
        \"Average\");
    printf(\"%-14.3f%-12.6f%-12.6f%-12.6f\\n\", Rate, Min, Max, Average);
").
