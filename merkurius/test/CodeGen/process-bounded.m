% A GPU process that lives through many messages in bounded memory
% (docs/INPROGRESS.md, Memory that a GPU process gives back, f3): an echo
% receives data(I, A), A an array of 1000 ints, makes an array of A's
% elements doubled, and answers back(I, Sum, B), Sum being A's sum. Each
% message it receives, with its array, and each array and answer it makes,
% are freed after their last use (Frees.cpp), so what it holds does not
% grow with the number of messages. The host sends 20000 messages, one
% after another, and checks each answer. The kind runs at the host here,
% and on the GPU in process-bounded-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_bounded.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

:- type to_echo
    --->    data(int, array(int))
    ;       stop.

:- type to_host
    --->    back(int, int, array(int)).

:- process echo(mailbox(to_host)) receives to_echo on gpu.

echo(Host, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(data(I, A)) then
        N = array.size(A),
        array.init(N, 0, B0),
        double(A, 0, N, B0, B),
        send(Host, back(I, sum(A, 0, N, 0), B), !IO),
        echo(Host, Self, !IO)
    else
        true
    ).

:- pred double(array(int)::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
double(A, I, N, !B) :-
    ( if I < N then
        !B ^ elem(I) := 2 * A ^ elem(I),
        double(A, I + 1, N, !B)
    else
        true
    ).

:- func sum(array(int), int, int, int) = int.
sum(A, I, N, S) = ( if I < N then sum(A, I + 1, N, S + A ^ elem(I)) else S ).

:- func size = int.
size = 1000.

:- func messages = int.
messages = 20000.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Me, !IO),
    new_mailbox(Downs, !IO),
    spawn(Place, echo(Me), E, !IO),
    monitor(E, Downs, id, _, !IO),
    exchange(E, Me, 0, 0, Wrong, !IO),
    io.format("%d messages, %d wrong\n", [i(messages), i(Wrong)], !IO),
    send(E, stop, !IO),
    receive(Downs, any, after(60000), _, !IO).

:- func id(down_info) = down_info.
id(D) = D.

    % Message I has the array of I + J for each index J; its answer's sum
    % is size * I + size * (size - 1) / 2, and the doubled array's last
    % element 2 * (I + size - 1).
:- pred exchange(mailbox(to_echo)::in, mailbox(to_host)::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
exchange(E, Me, I, !Wrong, !IO) :-
    ( if I < messages then
        array.init(size, 0, A0),
        fill(I, 0, A0, A),
        send(E, data(I, A), !IO),
        receive(Me, any, after(60000), R, !IO),
        ( if
            R = message(back(I, Sum, B)),
            Sum = size * I + size * (size - 1) / 2,
            array.size(B) = size,
            B ^ elem(size - 1) = 2 * (I + size - 1)
        then
            true
        else
            !:Wrong = !.Wrong + 1
        ),
        exchange(E, Me, I + 1, !Wrong, !IO)
    else
        true
    ).

:- pred fill(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
fill(I, J, !A) :-
    ( if J < size then
        !A ^ elem(J) := I + J,
        fill(I, J + 1, !A)
    else
        true
    ).

% CHECK:20000 messages, 0 wrong
