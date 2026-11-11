% Arrays between the host and a process kind that runs on a GPU. A summer
% takes three hundred arrays, of one to three hundred elements, and answers
% each with its sum and with an array it makes of its first and last
% elements; then a hundred summers take one each, the first thing in their
% arenas. Both copies of an array go through host memory, where the
% device copies them: a copy into or out of the device's own memory, made
% while the process runs, could meet lines of it that the device's L2
% cache holds, as it did, which reads gave old cells. The program runs its
% kind at the host here, and at the GPU in process-gpu-arrays-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_gpu_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process.

    % Answers each array with its sum to Sums and its ends to Ends, until
    % an empty one.
:- process summer(mailbox(int), mailbox(array(int))) receives array(int)
    on gpu.

summer(Sums, Ends, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(A), array.size(A) > 0 then
        send(Sums, sum(A, 0, 0), !IO),
        Last = array.size(A) - 1,
        array.init(2, array.lookup(A, 0), E0),
        array.set(1, array.lookup(A, Last), E0, E),
        send(Ends, E, !IO),
        summer(Sums, Ends, Self, !IO)
    else
        true
    ).

:- func sum(array(int), int, int) = int.

sum(A, I, S) =
    ( if I < array.size(A) then sum(A, I + 1, S + array.lookup(A, I))
    else S ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Sums, !IO),
    new_mailbox(Ends, !IO),
    spawn_summer_at(Place, Sums, Ends, S, !IO),
    rounds(S, Sums, Ends, 1, 300, 0, Wrong, !IO),
    send(S, array([]), !IO),
    io.write_int(Wrong, !IO),
    io.write_string(" wrong of 300\n", !IO),
    fresh(Place, Sums, Ends, 1, 100, 0, FreshWrong, !IO),
    io.write_int(FreshWrong, !IO),
    io.write_string(" wrong of 100\n", !IO).

    % A summer for each of N .. Max, which ends after one array.
:- pred fresh(place::in, mailbox(int)::in, mailbox(array(int))::in, int::in,
    int::in, int::in, int::out, io::di, io::uo) is det.

fresh(Place, Sums, Ends, N, Max, !Wrong, !IO) :-
    ( if N =< Max then
        spawn_summer_at(Place, Sums, Ends, S, !IO),
        rounds(S, Sums, Ends, N, N, !Wrong, !IO),
        send(S, array([]), !IO),
        fresh(Place, Sums, Ends, N + 1, Max, !Wrong, !IO)
    else
        true
    ).

    % Sends N .. N * 2 - 1, in N elements, and checks the answers.
:- pred rounds(mailbox(array(int))::in, mailbox(int)::in,
    mailbox(array(int))::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

rounds(S, Sums, Ends, N, Max, !Wrong, !IO) :-
    ( if N =< Max then
        send(S, array(N .. N * 2 - 1), !IO),
        receive(Sums, any, after(10000), Sum, !IO),
        receive(Ends, any, after(10000), End, !IO),
        Expected = N * (N * 3 - 1) / 2,
        ( if
            Sum = message(Expected),
            End = message(E),
            array.size(E) = 2,
            array.lookup(E, 0) = N,
            array.lookup(E, 1) = N * 2 - 1
        then
            true
        else
            !:Wrong = !.Wrong + 1,
            io.write_string("round ", !IO),
            io.write_int(N, !IO),
            io.write_string(": ", !IO),
            io.print(Sum, !IO), io.write_string(" ", !IO),
            io.print_line(End, !IO)
        ),
        rounds(S, Sums, Ends, N + 1, Max, !Wrong, !IO)
    else
        true
    ).
% CHECK:0 wrong of 300
% CHECK-NEXT:0 wrong of 100
