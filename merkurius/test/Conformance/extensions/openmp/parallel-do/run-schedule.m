% DESIGN: Parallel loop annotations
% A schedule clause deals a parallel loop's iterations out to the threads,
% static or dynamic, with a chunk size that is a positive integer or a
% variable bound before the directive; it changes which thread runs an
% iteration, not what the loop gives. The iterations here take uneven
% work, and each loop also counts its iterations, so one run twice or not
% at all shows.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/schedule_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{485730200, 1000}
% OUTPUT-NEXT:{485730200, 1000}
% OUTPUT-NEXT:{485730200, 1000}
% OUTPUT-NEXT:{485730200, 1000}
% OUTPUT-NOT:{{.}}
:- module run_schedule.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func work(int) = int.
work(I) = W :-
    ( for(_, 1, I mod 97), fromto(I, W0, W1, W) do W1 = (W0 * 31 + 7) mod 1000003 ).

:- pred static(int::in, int::out, int::out) is det.
static(N, S, C) :-
    %$omp parallel do schedule(static) reduction((+) : S) reduction((+) : C)
    ( for(I, 0, N - 1), fromto(0, S0, S1, S), fromto(0, C0, C1, C) do
        S1 = S0 + work(I), C1 = C0 + 1
    ).

:- pred static_chunks(int::in, int::out, int::out) is det.
static_chunks(N, S, C) :-
    %$omp parallel do schedule(static, 3) reduction((+) : S) reduction((+) : C)
    ( for(I, 0, N - 1), fromto(0, S0, S1, S), fromto(0, C0, C1, C) do
        S1 = S0 + work(I), C1 = C0 + 1
    ).

:- pred dynamic(int::in, int::out, int::out) is det.
dynamic(N, S, C) :-
    %$omp parallel do schedule(dynamic) reduction((+) : S) reduction((+) : C)
    ( for(I, 0, N - 1), fromto(0, S0, S1, S), fromto(0, C0, C1, C) do
        S1 = S0 + work(I), C1 = C0 + 1
    ).

:- pred dynamic_chunks(int::in, int::in, int::out, int::out) is det.
dynamic_chunks(N, K, S, C) :-
    %$omp parallel do schedule(dynamic, K) reduction((+) : S) reduction((+) : C)
    ( for(I, N - 1, 0, -1), fromto(0, S0, S1, S), fromto(0, C0, C1, C) do
        S1 = S0 + work(I), C1 = C0 + 1
    ).

main(!IO) :-
    static(1000, S1, C1), io.write({S1, C1}, !IO), io.nl(!IO),
    static_chunks(1000, S2, C2), io.write({S2, C2}, !IO), io.nl(!IO),
    dynamic(1000, S3, C3), io.write({S3, C3}, !IO), io.nl(!IO),
    dynamic_chunks(1000, 7, S4, C4), io.write({S4, C4}, !IO), io.nl(!IO).
