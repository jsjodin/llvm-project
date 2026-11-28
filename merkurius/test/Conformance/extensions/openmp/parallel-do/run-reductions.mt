% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% A value a parallel loop carries from one iteration to the next, a
% fromto's or a state variable's, is named in a reduction clause, by
% Last's name or the state variable's, or, the one scalar the loop
% carries, by none. First, or the state variable's value before the loop,
% is combined with the partial results by the clause's operator. Run on
% four threads, the loop gives what the sequential loop gives: its
% operator is associative, and the floats added here are exact, or
% printed to fewer digits than the order can change. It is a template, run
% for float and float32.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/reductions_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float.d/run_reductions_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float.d/run_reductions_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float.d/run_reductions_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float.d/run_reductions_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:3.14159265
% OUTPUT-FLOAT-NEXT:{6050, 10095}
% OUTPUT-FLOAT-NEXT:{3628800, -6, 6, 65535, -21846, 2396745, 10}
% OUTPUT-FLOAT-NEXT:{58.25, 0.5}
% OUTPUT-FLOAT-NEXT:{1000, -5}
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float32.d/run_reductions_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float32.d/run_reductions_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float32.d/run_reductions_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float32.d/run_reductions_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:3.14
% OUTPUT-FLOAT32-NEXT:{6050, 10095}
% OUTPUT-FLOAT32-NEXT:{3628800, -6, 6, 65535, -21846, 2396745, 10}
% OUTPUT-FLOAT32-NEXT:{58.25f32, 0.5f32}
% OUTPUT-FLOAT32-NEXT:{1000, -5}
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_reductions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

:- func pi(int) = @T@.
pi(N) = Step * Sum :-
    Step = 1.0@S@ / @FROM_INT@(N),
    %$omp parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0@S@, S0, S1, Sum) do
        X = (@FROM_INT@(I) + 0.5@S@) * Step,
        S1 = S0 + 4.0@S@ / (1.0@S@ + X * X)
    ).

:- pred sums(int::in, int::out, int::out) is det.
sums(N, A, S) :-
    %$omp parallel do reduction((+) : A)
    ( for(I, 1, N), fromto(1000, A0, A1, A) do A1 = A0 + I ),
    some [!T] (
        !:T = -5,
        %$omp parallel do reduction(+)
        ( for(I, 1, N) do !:T = !.T + 2 * I ),
        S = !.T
    ).

:- pred others(int::in, {int, int, int, int, int, int, int}::out) is det.
others(N, {P, Lo, Hi, X, And, Or, L}) :-
    %$omp parallel do reduction((*) : P)
    ( for(I, 1, N), fromto(1, P0, P1, P) do P1 = P0 * I ),
    %$omp parallel do reduction(min : Lo) reduction(max : Hi)
    ( for(I, 0, 2 * N), fromto(max_int, L0, L1, Lo), fromto(min_int, H0, H1, Hi) do
        V = (I * 7) mod 13 - 6,
        L1 = int.min(L0, V),
        H1 = int.max(H0, V)
    ),
    %$omp parallel do reduction(xor)
    ( for(I, 0, 15), fromto(0, X0, X1, X) do X1 = xor(X0, 1 << I) ),
    %$omp parallel do reduction((/\) : And)
    ( for(I, 0, 7), fromto(-1, A0, A1, And) do A1 = A0 /\ \ (1 << (2 * I)) ),
    %$omp parallel do reduction((\/) : Or)
    ( for(I, 0, 7), fromto(0, O0, O1, Or) do O1 = O0 \/ (1 << (3 * I)) ),
    %$omp parallel do reduction(int.max)
    ( for(I, 0, N - 1), fromto(min_int, M0, M1, L) do M1 = int.max(M0, (I * 5) mod 11) ).

:- pred halves(int::in, @T@::out, @T@::out) is det.
halves(N, Sum, Product) :-
    %$omp parallel do reduction((+) : Sum) reduction((*) : Product)
    ( for(I, 1, N), fromto(0.5@S@, S0, S1, Sum), fromto(1.0@S@, P0, P1, Product) do
        S1 = S0 + @FROM_INT@(I) / 4.0@S@,
        P1 = P0 * (if I mod 2 = 0 then 2.0@S@ else 0.5@S@)
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
