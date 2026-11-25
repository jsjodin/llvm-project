% DESIGN: Open terms (Prolog-style variables) / Checking rules
% A parallel do body may read an open value that its iterations share with
% from_open, and may bind open terms that each iteration makes on its own.
% Run on four threads and sequentially, the sums are the same.
% MERKC-FLAGS: -fopen-types -fopenmp -fdo-loops
% TRANSLATION: Inputs/parallel_do_translation.pl
%
% REQUIRES: libomp
% RUN: merkc -O0 -fopen-types -fopenmp -fdo-loops %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types -fopenmp -fdo-loops %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fopen-types -fdo-loops %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types -fdo-loops %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:unbound hole: 0
% OUTPUT-NEXT:bound hole: 6000
% OUTPUT-NEXT:own terms: 333833500
% OUTPUT-NOT:{{.}}
:- module run_parallel_do.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

    % Each iteration closes the shared list, once its hole is bound.
:- pred read_only(open(list(int))::ia, int::in, int::out) is det.

read_only(X, N, S) :-
    %$omp parallel do reduction((+) : S)
    ( for(_, 1, N), fromto(0, S0, S1, S) do
        promise_pure (
            if impure from_open(X, L) then
                S1 = S0 + list.foldl(int.plus, L, 0)
            else
                S1 = S0
        )
    ).

    % Each iteration makes [A, B], binds B to A and A to I, and closes it.
:- pred own_term(int::in, int::out) is det.

own_term(N, S) :-
    %$omp parallel do reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do
        promise_pure (
            L = [A, B] : open(list(int)),
            ( if
                A = B,
                A = to_open(I),
                impure from_open(L, [V, W])
            then
                S1 = S0 + V * W
            else
                S1 = S0
            )
        )
    ).

:- pred shared(open(list(int))::oa) is det.

shared(X) :-
    X = [to_open(1), to_open(2) | _].

main(!IO) :-
    shared(X),
    read_only(X, 1000, S1),
    io.format("unbound hole: %d\n", [i(S1)], !IO),
    promise_pure (
        if X = [_, _ | T], T = to_open([3]) then
            read_only(X, 1000, S2),
            io.format("bound hole: %d\n", [i(S2)], !IO)
        else
            io.write_string("could not bind the hole\n", !IO)
    ),
    own_term(1000, S3),
    io.format("own terms: %d\n", [i(S3)], !IO).
