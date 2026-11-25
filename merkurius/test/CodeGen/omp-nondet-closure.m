% A closure of a nondet or multi procedure, made in a parallel loop's body.
% Outlining rewrites such a procedure's closure entry to take the
% continuation and environment of its solutions, before the omp dialect's
% passes, which verify the module: merk.closure accepts that entry. Run on
% four threads, the program prints what the sequential program prints.
%
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_nondet_closure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, solutions.

:- pred even_member(list(int)::in, int::out) is nondet.
even_member(L, X) :-
    list.member(X, L),
    X mod 2 = 0.

:- pred pair_of(int::in, int::out) is multi.
pair_of(N, N).
pair_of(N, -N).

:- pred counts(int::in, array(int)::array_di, array(int)::array_uo) is det.
counts(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        solutions(even_member(1 .. N + I), Evens),
        solutions(pair_of(I), Pair),
        !A ^ unsafe_elem(I) := list.length(Evens) * 10 + list.foldl(plus, Pair, 0)
    ).

main(!IO) :-
    array.init(4, 0, A0),
    counts(5, A0, A),
    io.write_line(A, !IO).

% OUTPUT:array([20, 30, 30, 40])
% OUTPUT-NOT:{{.}}
