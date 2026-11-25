% DESIGN: Loops / Iterators
% for(I, Lo, Hi) takes the integers from Lo to Hi inclusive, and
% for(I, Lo, Hi, Step) those Step apart; a negative step counts down. The
% bounds are evaluated once, before the loop, and a loop whose Lo is past
% its Hi has no iterations. A loop to max_int ends.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/for_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 4, 9, 16, 25]
% OUTPUT-NEXT:[1, 4, 7, 10]
% OUTPUT-NEXT:[10, 6, 2]
% OUTPUT-NEXT:[]
% OUTPUT-NEXT:3
% OUTPUT-NOT:{{.}}
:- module run_for.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    ( for(I, 1, 5), foreach(X, Up) do X = I * I ),
    io.write(Up, !IO), io.nl(!IO),
    ( for(I, 1, 10, 3), foreach(X, Strided) do X = I ),
    io.write(Strided, !IO), io.nl(!IO),
    ( for(I, 10, 1, -4), foreach(X, Down) do X = I ),
    io.write(Down, !IO), io.nl(!IO),
    N = 3,
    ( for(I, N, N - 1), foreach(X, None) do X = I ),
    io.write(None, !IO), io.nl(!IO),
    ( for(I, max_int - 2, max_int), fromto(0, C0, C1, C) do C1 = C0 + 1, _ = I ),
    io.write_int(C, !IO), io.nl(!IO).
