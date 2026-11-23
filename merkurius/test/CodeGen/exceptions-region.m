% Exceptions in a parallel loop (docs/INPROGRESS.md, Exceptions). A try in the
% loop's body catches what its goal throws, on any thread of the team: a
% team thread that runs no process keeps its tries itself, and the thread
% that runs the process keeps them with the process's, made in the region.
% A throw that would leave the region, to a try made outside it, ends the
% program, as a team cannot be left by one thread (design document,
% "Exceptions, links and monitors"); without -fopenmp the loop is
% sequential, and the try catches it.
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %s -o %t.O2 && %libomp-env OMP_NUM_THREADS=4 %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: sed 's/^escape = no\./escape = yes./' %s > %t.e.m
% RUN: merkc -O2 -fopenmp -fdo-loops %libomp-flags %t.e.m -o %t.e && %libomp-env OMP_NUM_THREADS=4 not %t.e 2>&1 | FileCheck %s --check-prefix=LEAVES --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.e.m -o %t.e.sequential && %t.e.sequential | FileCheck %s --check-prefix=SEQUENTIAL --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module exceptions_region.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module array, bool, exception, int, list.

:- func escape = bool.
escape = no.

    % Throws for odd numbers, and for 77 always.
:- pred odd_throws(int::in, int::out) is det.

odd_throws(X, Y) :-
    L = list.map(func(K) = K + X, 1 .. 10),
    ( if (X mod 2 = 1 ; X = 77) then
        throw(X)
    else
        Y = list.foldl(int.plus, L, 0)
    ).

    % A try in each iteration: what the goal gives, or minus what it threw.
:- pred caught(int::in, int::out) is det.

caught(I, V) :-
    promise_equivalent_solutions [V] (
        try(odd_throws(I), R),
        (
            R = succeeded(V)
        ;
            R = exception(_),
            V = -I
        )
    ).

:- pred fill(array(int)::array_di, array(int)::array_uo) is det.

fill(!A) :-
    N = array.size(!.A),
    %$omp parallel do
    ( for(I, 0, N - 1) do
        caught(I, V),
        !A ^ unsafe_elem(I) := V
    ).

:- pred fill_escaping(array(int)::array_di, array(int)::array_uo) is det.

fill_escaping(!A) :-
    N = array.size(!.A),
    %$omp parallel do
    ( for(I, 0, N - 1) do
        ( if I = 77 then
            odd_throws(I, V)
        else
            V = I
        ),
        !A ^ unsafe_elem(I) := V
    ).

:- pred total(array(int)::in, int::in, int::in, int::out) is det.

total(A, I, S0, S) :-
    ( if I >= array.size(A) then
        S = S0
    else
        total(A, I + 1, S0 + A ^ elem(I), S)
    ).

main(!IO) :-
    Escape = escape,
    (
        Escape = no,
        array.init(200, 0, A0),
        fill(A0, A),
        total(A, 0, 0, S),
        % OUTPUT:94500
        io.write_int(S, !IO),
        io.nl(!IO)
    ;
        Escape = yes,
        array.init(200, 0, B0),
        try(fill_escaping(B0), R),
        (
            R = succeeded(B),
            total(B, 0, 0, S),
            io.write_int(S, !IO)
        ;
            % SEQUENTIAL:caught 77
            R = exception(_),
            io.write_string("caught 77", !IO)
        ),
        io.nl(!IO)
        % LEAVES:Uncaught Merkurius exception:
        % LEAVES-NEXT:77
    ).
