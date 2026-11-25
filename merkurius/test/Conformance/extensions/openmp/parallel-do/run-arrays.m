% DESIGN: Loops / Under a parallel directive
% A parallel loop's one iterator is a for, whose bounds are evaluated
% once, with any step, down or up, and none or one iteration. An array
% threaded through it, array_di to array_uo, is updated in place by every
% iteration and shared; variables bound before the loop are shared and
% read by every iteration, and those only in its body, bound in both
% branches of an if-then-else, are local to each. A loop in the body is
% sequential, inside each iteration, so its accumulator needs no
% reduction clause. The iterations write disjoint elements, as the
% directive promises, so the program prints what the sequential one does.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/arrays_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:0 10 0 0 40 0 0 70 0 0 100 0 0 130 0 0 160 0 0 190
% OUTPUT-NEXT:0 101 0 103 40 105 0 107 0 109 100 111 0 113 0 115 160 117 0 119
% OUTPUT-NEXT:0 101 0 103 40 99 0 107 0 109 100 111 0 113 0 115 160 117 0 119
% OUTPUT-NEXT:2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32
% OUTPUT-NOT:{{.}}
:- module run_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- pred fill(int::in, int::in, int::in, int::in, array(int)::array_di,
    array(int)::array_uo) is det.
fill(Lo, Hi, Step, K, !A) :-
    ( if Step > 0 then
        %$omp parallel do
        ( for(I, Lo, Hi, 3) do
            !A ^ unsafe_elem(I) := K * I
        )
    else
        %$omp parallel do
        ( for(I, Hi, Lo, -2) do
            ( if I mod 4 = 0 then V = -I else V = I + K ),
            !A ^ unsafe_elem(I) := V
        )
    ).

:- pred matmul(int::in, array(int)::in, array(int)::in,
    array(int)::array_di, array(int)::array_uo) is det.
matmul(N, A, B, !C) :-
    %$omp parallel do
    ( for(I, 0, N - 1) do
        ( for(J, 0, N - 1) do
            ( for(K, 0, N - 1), fromto(0, S0, S1, Sum) do
                S1 = S0 + A ^ unsafe_elem(I * N + K) * B ^ unsafe_elem(K * N + J)
            ),
            !C ^ unsafe_elem(I * N + J) := Sum
        )
    ).

:- pred once(int::in, array(int)::array_di, array(int)::array_uo) is det.
once(Lo, !A) :-
    %$omp parallel do
    ( for(I, Lo, Lo) do !A ^ unsafe_elem(I) := 99 ),
    %$omp parallel do
    ( for(I, Lo, Lo - 1) do !A ^ unsafe_elem(I) := -99 ).

:- pred show(array(int)::array_ui, io::di, io::uo) is det.
show(A, !IO) :-
    list.map(int_to_string, array.to_list(A), Strings),
    io.write_string(string.join_list(" ", Strings), !IO),
    io.nl(!IO).

main(!IO) :-
    array.init(20, 0, A0),
    fill(1, 19, 1, 10, A0, A1),
    show(A1, !IO),
    fill(0, 19, -1, 100, A1, A2),
    show(A2, !IO),
    once(5, A2, A3),
    show(A3, !IO),
    N = 4,
    M = array.generate(N * N, (func(X) = X + 1)),
    D = array.generate(N * N, (func(X) = (if X mod (N + 1) = 0 then 2 else 0))),
    array.init(N * N, 0, C0),
    matmul(N, M, D, C0, C),
    show(C, !IO).
