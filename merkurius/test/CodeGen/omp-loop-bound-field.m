% A parallel loop with a reduction whose bound reads a field of a record
% (P ^ rows): the bounds' goals are made before the region, for the
% reduction's first value, and again in it, as Flang places them. Made
% again, the field's deconstruction found its variable bound already, from
% before the region, and tested it against that: a guard, whose failure
% was a branch out of the parallel region before the loop, which the
% OpenMP dialect's verifier rejects for a combined parallel ("nested
% combined child op doesn't unconditionally execute"). The region now makes
% the bounds' goals afresh.
% - sum: parallel do, in an ordinary predicate.
% - spread: teams distribute parallel do in a GPU process's kind, which on
%   the host is a parallel do. The kind runs at the host here.
%
% The output is the sums of rows 2 to 4 and 1 to 5 of an 8 by 8 array of
% each cell's index: 16 + ... + 39 = 660, and 8 + ... + 47 = 1100.
%
% REQUIRES: libomp
% RUN: merkc -fc1 -emit-merk -fopenmp -fdo-loops -fprocess-declarations %s -o - | FileCheck %s
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations %libomp-flags %s -o %t
% RUN: env OMP_NUM_THREADS=4 %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations %libomp-flags %s -o %t.O2
% RUN: env OMP_NUM_THREADS=4 %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines

:- module omp_loop_bound_field.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type params
    --->    params(
                n :: int,
                first :: int,
                rows :: int
            ).

    % The cells of rows First to First + Rows - 1.
    %
:- pred sum(params::in, array(int)::array_ui, int::out) is det.
% CHECK-LABEL: merk.proc private @omp_loop_bound_field.sum_3_0(
% CHECK:         merk.deconstruct #merk.functor<"params", 3>
% CHECK:         omp.parallel
% CHECK-NEXT:      arith.muli
% CHECK-NEXT:      merk.deconstruct #merk.functor<"params", 3>
% CHECK-NOT:       merk.guard
% CHECK:           omp.wsloop
sum(P, A, S) :-
    N = P ^ n,
    First = P ^ first,
    %$omp parallel do reduction((+) : S)
    ( for(K, First * N, (First + P ^ rows) * N - 1), fromto(0, S0, S1, S) do
        S1 = S0 + A ^ unsafe_elem(K)
    ).

    % Answers a number of rows R with the sum of rows 1 to R.
    %
:- process spreader(mailbox(int)) receives int on gpu.

spreader(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(Rows) then
        indices(A),
        spread(params(8, 1, Rows), A, S),
        send(Reply, S, !IO)
    else
        true
    ).

:- pred spread(params::in, array(int)::array_ui, int::out) is det.
% CHECK-LABEL: merk.proc private @omp_loop_bound_field.spread_3_0(
% CHECK:         omp.parallel
% CHECK-NOT:       merk.guard
% CHECK:           omp.wsloop
spread(P, A, S) :-
    N = P ^ n,
    First = P ^ first,
    %$omp teams distribute parallel do reduction((+) : S)
    ( for(K, First * N, (First + P ^ rows) * N - 1), fromto(0, S0, S1, S) do
        S1 = S0 + A ^ unsafe_elem(K)
    ).

    % An 8 by 8 array of each cell's index.
    %
:- pred indices(array(int)::array_uo) is det.
indices(A) :-
    array.init(64, 0, A0),
    ( for(I, 0, 63), fromto(A0, B0, B1, A) do
        B1 = B0 ^ elem(I) := I
    ).

main(!IO) :-
    indices(A),
    sum(params(8, 2, 3), A, S),
    io.format("sum: %d\n", [i(S)], !IO),
    new_mailbox(Reply, !IO),
    spawn(host, spreader(Reply), M, !IO),
    send(M, 5, !IO),
    receive(Reply, any, after(60000), R, !IO),
    ( if R = message(T) then
        io.format("spread: %d\n", [i(T)], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).

% OUTPUT:sum: 660
% OUTPUT-NEXT:spread: 1100
