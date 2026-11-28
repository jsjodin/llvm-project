% The math library's functions in a GPU loop (benchmarks/math/README.md):
% each function's loop over MATH_N elements, MATH_K applications of it an
% element, under one OpenMP target region a function, timed after a run
% that loads the kernel. Prints each function's name, the best of
% MATH_REPEAT runs' seconds and the sum of the elements' last values.
:- module math_loop.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module float, int, list, math, string.

main(!IO) :-
    setting("MATH_N", 1 << 20, N, !IO),
    setting("MATH_K", 1000, K, !IO),
    time("sqrt", run_sqrt, N, K, !IO),
    time("exp", run_exp, N, K, !IO),
    time("ln", run_ln, N, K, !IO),
    time("sin", run_sin, N, K, !IO),
    time("cos", run_cos, N, K, !IO),
    time("pow", run_pow, N, K, !IO),
    time("tanh", run_tanh, N, K, !IO).

    % The best of MATH_REPEAT runs of the function's loop, after one that
    % loads its kernel.
:- pred time(string::in, (func(int, int) = float)::in, int::in, int::in,
    io::di, io::uo) is det.
time(Name, Run, N, K, !IO) :-
    setting("MATH_REPEAT", 5, Repeat, !IO),
    First = Run(N, K),
    best(Repeat, Run, N, K, 1.0e30, Best, !IO),
    io.format("%-6s %10.6f %.17g\n", [s(Name), f(Best), f(First)], !IO).

:- pred best(int::in, (func(int, int) = float)::in, int::in, int::in,
    float::in, float::out, io::di, io::uo) is det.
best(Left, Run, N, K, Best0, Best, !IO) :-
    ( if Left =< 0 then
        Best = Best0
    else
        now(T0, !IO),
        Sum = Run(N, K),
        keep(Sum, !IO),
        now(T1, !IO),
        best(Left - 1, Run, N, K, float.min(Best0, T1 - T0), Best, !IO)
    ).

:- func run_sqrt(int, int) = float.
run_sqrt(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.sqrt(X0 + C)
        ),
        S1 = S0 + X
    ).

:- func run_exp(int, int) = float.
run_exp(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.exp(-(X0 + C))
        ),
        S1 = S0 + X
    ).

:- func run_ln(int, int) = float.
run_ln(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.ln(X0 + 2.0 + C)
        ),
        S1 = S0 + X
    ).

:- func run_sin(int, int) = float.
run_sin(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.sin(X0 + 1.0 + C)
        ),
        S1 = S0 + X
    ).

:- func run_cos(int, int) = float.
run_cos(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.cos(X0 + C)
        ),
        S1 = S0 + X
    ).

:- func run_pow(int, int) = float.
run_pow(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.pow(X0 + 1.0 + C, 0.75)
        ),
        S1 = S0 + X
    ).

:- func run_tanh(int, int) = float.
run_tanh(N, K) = Sum :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        C = float(I) / float(N),
        ( for(_, 1, K), fromto(C, X0, X1, X) do
            X1 = math.tanh(X0 + 0.5 + C)
        ),
        S1 = S0 + X
    ).

    % Uses the sum, so that the run is not optimized away.
:- pred keep(float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    keep(S::in, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    volatile double sink = S;
    (void) sink;
").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

    % The environment variable's value, if it is set, and otherwise Default.
:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

:- pragma foreign_decl("C", "
#include <stdlib.h>
#include <time.h>
").
