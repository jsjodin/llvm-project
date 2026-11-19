% Loops across a GPU process's teams in predicates that nothing reaches
% (docs/INPROGRESS.md, Spawn-time specialization of GPU processes): one in
% a predicate called only in a branch whose condition cannot succeed, as a
% program specialized on its constants has, and one in a predicate that
% nothing calls. Neither is an error, as a loop across the teams in a
% predicate that main reaches is (process-loops-errors.m), and the program
% runs the branch that can. The kind runs at the host here, and on the GPU
% in process-loops-unreached-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: libomp
% RUN: merkc -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t 2>&1 | FileCheck %s --check-prefix=WARN --allow-empty
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
%
% WARN-NOT: error

:- module process_loops_unreached.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- process squares(mailbox(int)) receives int on gpu.

squares(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        array.init(N, 0, A0),
        % The kernel number, as a specializer substitutes it.
        ( if 2 = 3 then
            cubes(N, A0, A)
        else
            square_all(N, A0, A)
        ),
        send(Reply, array.lookup(A, N - 1), !IO)
    else
        true
    ).

:- pred square_all(int::in, array(int)::array_di, array(int)::array_uo) is det.
square_all(N, !A) :-
    %$omp teams distribute parallel do
    ( for(I, 0, N - 1), fromto(!.A, A1, A2, !:A) do
        array.unsafe_set(I, I * I, A1, A2)
    ).

:- pred cubes(int::in, array(int)::array_di, array(int)::array_uo) is det.
cubes(N, !A) :-
    %$omp teams distribute parallel do
    ( for(I, 0, N - 1), fromto(!.A, A1, A2, !:A) do
        array.unsafe_set(I, I * I * I, A1, A2)
    ).

:- pred unused(int::in, array(int)::array_di, array(int)::array_uo) is det.
:- pragma no_inline(unused/3).
unused(N, !A) :-
    %$omp teams distribute
    ( for(I, 0, N - 1), fromto(!.A, A1, A2, !:A) do
        array.unsafe_set(I, -I, A1, A2)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Reply, !IO),
    spawn(Place, squares(Reply), Squares, !IO),
    send(Squares, 1000, !IO),
    receive(Reply, any, after(60000), R, !IO),
    ( if R = message(Last) then
        io.format("last square: %d\n", [i(Last)], !IO)
    else
        io.write_string("no answer\n", !IO)
    ).
% CHECK:last square: 998001
