% MANUAL: Determinism categories
% DESIGN: Loops / Under a parallel directive
% Interaction: nondet and semidet code, in a parallel loop's iterations,
% which make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/parallel_do_determinism_translation.m
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
% OUTPUT:1: evens 2, first over 2 3 | evens 2,4, first over 2 3 | evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3
% OUTPUT-NEXT:2: evens 2,4, first over 2 3 | evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4,6, first over 2 3
% OUTPUT-NEXT:3: evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3
% OUTPUT-NEXT:4: evens 2,4,6, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3 | evens 2,4,6,8, first over 2 3
% OUTPUT-NOT:{{.}}
:- module parallel_do_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, solutions, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = 1 .. N + 2.

use(L) = S :-
    solutions(even_member(L), Evens),
    ( if first_over(L, 2, F) then
        First = string.int_to_string(F)
    else
        First = "none"
    ),
    S = "evens " ++ string.join_list(",",
        list.map(string.int_to_string, Evens)) ++ ", first over 2 " ++ First.

:- pred even_member(list(int)::in, int::out) is nondet.

even_member(L, X) :-
    list.member(X, L),
    X mod 2 = 0.

:- pred first_over(list(int)::in, int::in, int::out) is semidet.

first_over([X | Xs], Limit, F) :-
    ( if X > Limit then
        F = X
    else
        first_over(Xs, Limit, F)
    ).

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make(N + I),
        !A ^ unsafe_elem(I) := use(V)
    ).

:- func run(int) = string.

run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
