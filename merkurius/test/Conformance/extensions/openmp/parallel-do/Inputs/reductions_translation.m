% The translation of run-reductions.m's loops, without their directives.
:- module reductions_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, string.

:- func pi(int) = float.
pi(N) = Step * Sum :-
    Step = 1.0 / float(N),
    pi_loop(0, N - 1, Step, 0.0, Sum).

:- pred pi_loop(int::in, int::in, float::in, float::in, float::out) is det.
pi_loop(I, Hi, Step, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        X = (float(I) + 0.5) * Step,
        S1 = S0 + 4.0 / (1.0 + X * X),
        pi_loop(I + 1, Hi, Step, S1, Sum)
    ).

% for(I, Lo, Hi) with one int accumulator updated by F.
:- pred fold(int::in, int::in, (func(int, int) = int)::in, int::in,
    int::out) is det.
fold(I, Hi, F, A0, A) :-
    ( if I > Hi then
        A = A0
    else
        fold(I + 1, Hi, F, F(I, A0), A)
    ).

:- pred sums(int::in, int::out, int::out) is det.
sums(N, A, S) :-
    fold(1, N, (func(I, A0) = A0 + I), 1000, A),
    fold(1, N, (func(I, T0) = T0 + 2 * I), -5, S).

:- pred others(int::in, {int, int, int, int, int, int, int}::out) is det.
others(N, {P, Lo, Hi, X, And, Or, L}) :-
    fold(1, N, (func(I, P0) = P0 * I), 1, P),
    fold(0, 2 * N, (func(I, L0) = int.min(L0, (I * 7) mod 13 - 6)),
        max_int, Lo),
    fold(0, 2 * N, (func(I, H0) = int.max(H0, (I * 7) mod 13 - 6)),
        min_int, Hi),
    fold(0, 15, (func(I, X0) = xor(X0, 1 << I)), 0, X),
    fold(0, 7, (func(I, A0) = A0 /\ \ (1 << (2 * I))), -1, And),
    fold(0, 7, (func(I, O0) = O0 \/ (1 << (3 * I))), 0, Or),
    fold(0, N - 1, (func(I, M0) = int.max(M0, (I * 5) mod 11)), min_int, L).

:- pred halves(int::in, float::out, float::out) is det.
halves(N, Sum, Product) :-
    halves_loop(1, N, 0.5, Sum, 1.0, Product).

:- pred halves_loop(int::in, int::in, float::in, float::out, float::in,
    float::out) is det.
halves_loop(I, Hi, S0, Sum, P0, Product) :-
    ( if I > Hi then
        Sum = S0, Product = P0
    else
        S1 = S0 + float(I) / 4.0,
        P1 = P0 * (if I mod 2 = 0 then 2.0 else 0.5),
        halves_loop(I + 1, Hi, S1, Sum, P1, Product)
    ).

main(!IO) :-
    io.write_string(string.format("%.8f\n", [f(pi(100000))]), !IO),
    sums(100, A, S),
    io.write({A, S}, !IO), io.nl(!IO),
    others(10, Others),
    io.write(Others, !IO), io.nl(!IO),
    halves(21, Sum, Product),
    io.write({Sum, Product}, !IO), io.nl(!IO),
    sums(0, A0, S0),
    io.write({A0, S0}, !IO), io.nl(!IO).
