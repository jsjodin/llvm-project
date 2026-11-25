% DESIGN: Loops / Under a parallel directive
% A target teams distribute parallel do loop runs its iterations across a
% device's teams and their threads. A value carried from one iteration to
% the next is named in a reduction clause, and its partial results are
% combined across threads and teams; an array threaded through the loop is
% updated in place, each iteration its own elements, and comes back.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/teams_loop_translation.m
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
% OUTPUT:{20, -50, 50}
% OUTPUT-NEXT:1.0
% OUTPUT-NEXT:1.0
% OUTPUT-NOT:{{.}}
:- module run_teams_loop.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- pred stats(int::in, int::out, int::out, int::out) is det.
stats(N, Sum, Lo, Hi) :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    %$omp& reduction(min : Lo) reduction(max : Hi)
    ( for(I, 1, N), fromto(0, S0, S1, Sum), fromto(max_int, L0, L1, Lo),
        fromto(min_int, H0, H1, Hi) do
        V = (I * 37) mod 101 - 50,
        S1 = S0 + V,
        L1 = int.min(L0, V),
        H1 = int.max(H0, V)
    ).

:- pred normalize(array(float)::array_di, array(float)::array_uo) is det.
normalize(!A) :-
    N = array.size(!.A),
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0, S0, S1, Sum) do
        S1 = S0 + !.A ^ elem(I)
    ),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ elem(I) := !.A ^ elem(I) / Sum
    ).

main(!IO) :-
    stats(100000, Sum, Lo, Hi),
    io.write({Sum, Lo, Hi}, !IO), io.nl(!IO),
    array.init(4096, 0.5, A0),
    normalize(A0, A),
    io.write_float(A ^ elem(0) * 4096.0, !IO), io.nl(!IO),
    io.write_float(A ^ elem(4095) * 4096.0, !IO), io.nl(!IO).
