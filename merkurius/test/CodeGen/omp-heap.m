% Each thread allocates on a heap of its own (runtime/heap.h), so the
% iterations of a parallel loop can make terms at once: each builds a list
% of its own, and sums it. Run on four threads, the program prints what the
% sequential program prints.
%
% RUN: merkc -fopenmp -fdo-loops %libomp-flags %s -o %t && %libomp-env OMP_NUM_THREADS=4 %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -fdo-loops %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_heap.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list.

:- func upto(int, list(int)) = list(int).
upto(N, L) = ( if N = 0 then L else upto(N - 1, [N | L]) ).

:- func sum(list(int), int) = int.
sum([], S) = S.
sum([X | Xs], S) = sum(Xs, S + X).

% Lists of up to 3 MB, many chunks each.
:- pred sums(array(int)::array_di, array(int)::array_uo) is det.
sums(!A) :-
    N = array.size(!.A),
    %$omp parallel do
    ( for(I, 0, N - 1) do
        !A ^ unsafe_elem(I) := sum(upto(I * 1000, []), 0)
    ).

:- pred total(array(int)::in, int::in, int::in, int::out) is det.
total(A, I, S0, S) :-
    ( if I >= array.size(A) then
        S = S0
    else
        total(A, I + 1, S0 + A ^ elem(I), S)
    ).

main(!IO) :-
    array.init(200, 0, A0),
    sums(A0, A),
    io.write_int(A ^ elem(199), !IO),
    io.nl(!IO),
    total(A, 0, 0, S),
    io.write_int(S, !IO),
    io.nl(!IO).
% OUTPUT:19800599500
% OUTPUT-NEXT:1323359950000
