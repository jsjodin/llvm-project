% A memory bandwidth benchmark modeled on BabelStream (Deakin et
% al., "Evaluating attainable memory bandwidth of parallel programming
% models via BabelStream", 2018): five kernels over three arrays of
% floats, each run a number of times, and the bandwidth each attains from
% its fastest run, after a check of the arrays' final values.
%
%   Copy    C = A
%   Mul     B = Scalar * C
%   Add     C = A + B
%   Triad   A = B + Scalar * C
%   Dot     Sum = A . B
%
% Each kernel is a do loop under %$omp target teams distribute parallel
% do, so the one program runs on a GPU when built with --offload-arch, on
% the host's threads with -fopenmp alone, as OpenMP runs a target region
% on the host when there is no device, and sequentially without it, when
% the directives are comments. See README.md.
%
% BANDWIDTH_ARRAY_SIZE and BANDWIDTH_NUM_TIMES in the environment set
% the arrays' length and the number of times, 2^25 and 100 by default.

:- module bandwidth.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- func start_a = float.
start_a = 0.1.
:- func start_b = float.
start_b = 0.2.
:- func start_c = float.
start_c = 0.0.
:- func scalar = float.
scalar = 0.4.

main(!IO) :-
    setting("BANDWIDTH_ARRAY_SIZE", 33554432, N, !IO),
    setting("BANDWIDTH_NUM_TIMES", 100, Times, !IO),
    ( if N < 1 then
        usage(!IO)
    else if Times < 2 then
        usage(!IO)
    else
        some [!A, !B, !C] (
            header(N, Times, !IO),
            init(N, !:A, !:B, !:C),
            % Each kernel's time in each run, kernel by kernel.
            array.init(kernels * Times, 0.0, T0),
            % The arrays stay on the device for the runs, so that each
            % kernel measures its memory and not the copies to it.
            %$omp target data map(tofrom: A, B, C)
            run(0, Times, N, !A, !B, !C, T0, T, 0.0, Sum, !IO),
            check(N, Times, !.A, !.B, !.C, Sum, Valid, !IO),
            ( if Valid = 1 then
                table_header(!IO),
                Word = 8.0,
                Size = float(N) * Word,
                row("Copy", 0, 2.0 * Size, Times, T, !IO),
                row("Mul", 1, 2.0 * Size, Times, T, !IO),
                row("Add", 2, 3.0 * Size, Times, T, !IO),
                row("Triad", 3, 3.0 * Size, Times, T, !IO),
                row("Dot", 4, 2.0 * Size, Times, T, !IO)
            else
                exit_failure(!IO)
            )
        )
    ).

:- pred usage(io::di, io::uo) is det.
usage(!IO) :-
    io.write_string("BANDWIDTH_ARRAY_SIZE must be at least 1 and BANDWIDTH_NUM_TIMES at least 2.\n", !IO),
    exit_failure(!IO).

:- func kernels = int.
kernels = 5.

% The arrays, each element its start value, set by a kernel too, so that,
% on the host, each thread first touches the elements it uses.
:- pred init(int::in, array(float)::array_uo, array(float)::array_uo,
    array(float)::array_uo) is det.
init(N, !:A, !:B, !:C) :-
    array.init(N, 0.0, !:A),
    array.init(N, 0.0, !:B),
    array.init(N, 0.0, !:C),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := start_a,
        !B ^ unsafe_elem(I) := start_b,
        !C ^ unsafe_elem(I) := start_c
    ).

%---------------------------------------------------------------------------%
% The kernels.

% Each reads its other arrays, which stay unique for the kernels after.
:- pred copy(int::in, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo) is det.
copy(N, A, !C) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !C ^ unsafe_elem(I) := A ^ unsafe_elem(I)
    ).

:- pred mul(int::in, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo) is det.
mul(N, C, !B) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !B ^ unsafe_elem(I) := scalar * C ^ unsafe_elem(I)
    ).

:- pred add(int::in, array(float)::array_ui, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo) is det.
add(N, A, B, !C) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !C ^ unsafe_elem(I) := A ^ unsafe_elem(I) + B ^ unsafe_elem(I)
    ).

:- pred triad(int::in, array(float)::array_ui, array(float)::array_ui,
    array(float)::array_di, array(float)::array_uo) is det.
triad(N, B, C, !A) :-
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := B ^ unsafe_elem(I) + scalar * C ^ unsafe_elem(I)
    ).

:- func dot(int, array(float), array(float)) = float.
:- mode dot(in, array_ui, array_ui) = out is det.
dot(N, A, B) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + A ^ unsafe_elem(I) * B ^ unsafe_elem(I)
    ).

%---------------------------------------------------------------------------%
% The runs, each kernel timed on its own.

:- pred run(int::in, int::in, int::in,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    float::in, float::out, io::di, io::uo) is det.
run(K, Times, N, !A, !B, !C, !T, !Sum, !IO) :-
    ( if K >= Times then
        true
    else
        now(T0, !IO),
        copy(N, !.A, !C),
        now(T1, !IO),
        mul(N, !.C, !B),
        now(T2, !IO),
        add(N, !.A, !.B, !C),
        now(T3, !IO),
        triad(N, !.B, !.C, !A),
        now(T4, !IO),
        !:Sum = dot(N, !.A, !.B),
        now(T5, !IO),
        !T ^ elem(0 * Times + K) := T1 - T0,
        !T ^ elem(1 * Times + K) := T2 - T1,
        !T ^ elem(2 * Times + K) := T3 - T2,
        !T ^ elem(3 * Times + K) := T4 - T3,
        !T ^ elem(4 * Times + K) := T5 - T4,
        run(K + 1, Times, N, !A, !B, !C, !T, !Sum, !IO)
    ).

%---------------------------------------------------------------------------%
% The check, as BabelStream's: the kernels on scalars give the values every
% element should have; the average error of each array, relative to its
% value, must be within 100 epsilon, and the dot product's within 1e-8.
% Valid is 1 if they all are, and 0 otherwise.

:- pred check(int::in, int::in, array(float)::in, array(float)::in,
    array(float)::in, float::in, int::out, io::di, io::uo) is det.
check(N, Times, A, B, C, Sum, Valid, !IO) :-
    gold(Times, start_a, GoldA, start_b, GoldB, start_c, GoldC),
    GoldSum = GoldA * GoldB * float(N),
    ErrA = error(N, A, GoldA),
    ErrB = error(N, B, GoldB),
    ErrC = error(N, C, GoldC),
    ErrSum = float.abs((Sum - GoldSum) / GoldSum),
    Epsilon = 100.0 * float.epsilon,
    check_array("a", ErrA, GoldA, Epsilon, ValidA, !IO),
    check_array("b", ErrB, GoldB, Epsilon, ValidB, !IO),
    check_array("c", ErrC, GoldC, Epsilon, ValidC, !IO),
    ( if ErrSum > 1.0e-8 then
        validation_sum(Sum, GoldSum, !IO),
        ValidSum = 0
    else
        ValidSum = 1
    ),
    Valid = ValidA * ValidB * ValidC * ValidSum.

:- pred gold(int::in, float::in, float::out, float::in, float::out,
    float::in, float::out) is det.
gold(K, A0, A, B0, B, C0, C) :-
    ( if K = 0 then
        A = A0, B = B0, C = C0
    else
        C1 = A0,
        B1 = scalar * C1,
        C2 = A0 + B1,
        A1 = B1 + scalar * C2,
        gold(K - 1, A1, A, B1, B, C2, C)
    ).

% The average of the elements' absolute errors.
:- func error(int, array(float), float) = float.
error(N, X, Gold) = Sum / float(N) :-
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + float.abs(X ^ unsafe_elem(I) - Gold)
    ).

:- pred check_array(string::in, float::in, float::in, float::in, int::out,
    io::di, io::uo) is det.
check_array(Name, Err, Gold, Epsilon, Valid, !IO) :-
    ( if Err / Gold > Epsilon then
        validation_array(Name, Err, !IO),
        Valid = 0
    else
        Valid = 1
    ).

%---------------------------------------------------------------------------%
% A kernel's row: its bandwidth from its fastest run, and its fastest,
% slowest and average times, leaving out the first run, as BabelStream
% does, which includes the device's start.

:- pred row(string::in, int::in, float::in, int::in, array(float)::in,
    io::di, io::uo) is det.
row(Name, Kernel, Bytes, Times, T, !IO) :-
    stats(Kernel * Times + 1, Kernel * Times + Times - 1, T,
        float.max, Min, 0.0, Max, 0.0, Total),
    Average = Total / float(Times - 1),
    write_row(Name, 1.0e-6 * Bytes / Min, Min, Max, Average, !IO).

:- pred stats(int::in, int::in, array(float)::in, float::in, float::out,
    float::in, float::out, float::in, float::out) is det.
stats(From, To, T, !Min, !Max, !Total) :-
    ( for(I, From, To) do
        X = T ^ elem(I),
        !:Min = float.min(!.Min, X),
        !:Max = float.max(!.Max, X),
        !:Total = !.Total + X
    ).

%---------------------------------------------------------------------------%
% What C does for the program: the clock, the environment and printf.

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

% The environment variable's value, if it is set, and otherwise Default.
:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

% Exits with status 1, since io.set_exit_status is not implemented yet.
:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").

:- pred header(int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(N::in, Times::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    double mb = 1.0e-6 * 8.0 * (double) N;
    printf(\"Merkurius memory bandwidth benchmark\\n\");
    printf(\"Running kernels %ld times\\n\", (long) Times);
    printf(\"Precision: double\\n\");
    printf(\"Array size: %.1f MB (=%.1f GB)\\n\", mb, 1.0e-3 * mb);
    printf(\"Total size: %.1f MB (=%.1f GB)\\n\", 3.0 * mb, 3.0e-3 * mb);
").

:- pred table_header(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    table_header(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-12s%-12s%-12s%-12s%-12s\\n\",
        \"Function\", \"MBytes/sec\", \"Min (sec)\", \"Max\", \"Average\");
").

:- pred write_row(string::in, float::in, float::in, float::in, float::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_row(Name::in, MBytes::in, Min::in, Max::in, Average::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-12s%-12.3f%-12.5f%-12.5f%-12.5f\\n\",
        Name, MBytes, Min, Max, Average);
").

:- pred validation_array(string::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    validation_array(Name::in, Err::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Validation failed on %s[]. Average error %g\\n\", Name, Err);
").

:- pred validation_sum(float::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    validation_sum(Sum::in, Gold::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Validation failed on sum. Error %g\\nSum was %.17g but should be %.17g\\n\",
        (Sum - Gold) / Gold, Sum, Gold);
").
