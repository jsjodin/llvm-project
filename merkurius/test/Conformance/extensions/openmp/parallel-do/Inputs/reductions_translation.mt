% The translation of run-reductions.mt's loops, without their directives,
% a template as the test is.
:- module reductions_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

:- func pi(int) = @T@.
pi(N) = Step * Sum :-
    Step = 1.0@S@ / @FROM_INT@(N),
    pi_loop(0, N - 1, Step, 0.0@S@, Sum).

:- pred pi_loop(int::in, int::in, @T@::in, @T@::in, @T@::out) is det.
pi_loop(I, Hi, Step, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        X = (@FROM_INT@(I) + 0.5@S@) * Step,
        S1 = S0 + 4.0@S@ / (1.0@S@ + X * X),
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

:- pred halves(int::in, @T@::out, @T@::out) is det.
halves(N, Sum, Product) :-
    halves_loop(1, N, 0.5@S@, Sum, 1.0@S@, Product).

:- pred halves_loop(int::in, int::in, @T@::in, @T@::out, @T@::in,
    @T@::out) is det.
halves_loop(I, Hi, S0, Sum, P0, Product) :-
    ( if I > Hi then
        Sum = S0, Product = P0
    else
        S1 = S0 + @FROM_INT@(I) / 4.0@S@,
        P1 = P0 * (if I mod 2 = 0 then 2.0@S@ else 0.5@S@),
        halves_loop(I + 1, Hi, S1, Sum, P1, Product)
    ).

    % The terms of the pi sum, and the digits it is printed to, that no
    % order of the threads' additions changes: fewer for float32, whose
    % sum's rounding errors grow faster.
:- func pi_precision = {int, int}.
pi_precision =
    ( if @T@.mantissa_digits > 24 then {100000, 8} else {1000, 2} ).

main(!IO) :-
    {Terms, Digits} = pi_precision,
    io.write_string(string.format("%." ++ string.int_to_string(Digits) ++
        "f\n", [f(@TO_FLOAT@(pi(Terms)))]), !IO),
    sums(100, A, S),
    io.write({A, S}, !IO), io.nl(!IO),
    others(10, Others),
    io.write(Others, !IO), io.nl(!IO),
    halves(21, Sum, Product),
    io.write({Sum, Product}, !IO), io.nl(!IO),
    sums(0, A0, S0),
    io.write({A0, S0}, !IO), io.nl(!IO).
