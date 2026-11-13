% Messages of a fixed part followed by one block, between the host and
% process kinds that run on a GPU (design document, Messages and memory on
% the device): each constructor's fields are of a fixed size, but for its
% last, which may be an array of scalars or a string. A doubler takes
% `cells(Step, Tile, Cells)`, makes an array of each cell doubled and sends
% it on to a summer as `doubled(Step, Tile, Doubled)`; the summer answers
% the host with `total(Step, Sum, Ends)`, Ends an array it makes of the
% first and last cells, and each `named(Step, Name)` with `echo(Step, Name)`.
% The arrays have from one to two hundred elements. Everything goes through
% host memory, copied by the message type's type_info. The kinds run at the
% host here, and on the GPU in process-block-messages-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: each answer is checked against what the host computes.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_block_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, int32, list, process, string, uint8.

:- type to_doubler
    --->    cells(int32, int, array(uint8))
    ;       stop_doubler.

:- type to_summer
    --->    doubled(int32, int, array(int))
    ;       named(int, string)
    ;       stop_summer.

:- type to_host
    --->    total(int32, int, array(int))
    ;       echo(int, string)
    ;       confused.

:- process doubler(mailbox(to_summer)) receives to_doubler on gpu.

doubler(Summer, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(cells(Step, Tile, Cells)) then
        N = array.size(Cells),
        array.init(N, 0, D0),
        double(Cells, 0, N, D0, D),
        send(Summer, doubled(Step, Tile, D), !IO),
        doubler(Summer, Self, !IO)
    else
        send(Summer, stop_summer, !IO)
    ).

:- pred double(array(uint8)::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
double(Cells, I, N, !D) :-
    ( if I < N then
        !D ^ elem(I) := 2 * uint8.to_int(Cells ^ elem(I)),
        double(Cells, I + 1, N, !D)
    else
        true
    ).

:- process summer(mailbox(to_host)) receives to_summer on gpu.

summer(Host, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(doubled(Step, Tile, D)) then
        N = array.size(D),
        array.init(2, D ^ elem(0), E0),
        array.set(1, D ^ elem(N - 1), E0, E),
        send(Host, total(Step, Tile + sum(D, 0, N, 0), E), !IO),
        summer(Host, Self, !IO)
    else if R = message(named(Step, Name)) then
        send(Host, echo(Step + 1, Name), !IO),
        summer(Host, Self, !IO)
    else if R = message(stop_summer) then
        true
    else
        send(Host, confused, !IO)
    ).

:- func sum(array(int), int, int, int) = int.
sum(D, I, N, S) = ( if I < N then sum(D, I + 1, N, S + D ^ elem(I)) else S ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["host"] then host else gpu(0) ),
    new_mailbox(Host, !IO),
    spawn(Place, summer(Host), Summer, !IO),
    spawn(Place, doubler(Summer), Doubler, !IO),
    rounds(1, 200, Doubler, Summer, Host, 0, Wrong, !IO),
    io.format("%d wrong of 200\n", [i(Wrong)], !IO),
    send(Summer, named(41, "a string after the arrays"), !IO),
    receive(Host, any, after(60000), R, !IO),
    ( if R = message(echo(Step, Name)) then
        io.format("%d %s\n", [i(Step), s(Name)], !IO)
    else
        io.write_string("no echo\n", !IO)
    ),
    send(Doubler, stop_doubler, !IO).

    % Round K sends K cells, cell I being (K + I) mod 256, and checks the
    % answer.
:- pred rounds(int::in, int::in, mailbox(to_doubler)::in,
    mailbox(to_summer)::in, mailbox(to_host)::in, int::in, int::out,
    io::di, io::uo) is det.
rounds(K, Last, Doubler, Summer, Host, !Wrong, !IO) :-
    ( if K > Last then
        true
    else
        Cells = array.generate(K,
            (func(I) = uint8.cast_from_int((K + I) mod 256))),
        send(Doubler, cells(int32.cast_from_int(K), 1000 * K, Cells), !IO),
        receive(Host, any, after(60000), R, !IO),
        Expected = 1000 * K + 2 * expected_sum(K, 0, 0),
        First = 2 * (K mod 256),
        Last1 = 2 * ((2 * K - 1) mod 256),
        ( if
            R = message(total(Step, Sum, E)),
            Step = int32.cast_from_int(K),
            Sum = Expected,
            array.to_list(E) = [First, Last1]
        then
            true
        else
            !:Wrong = !.Wrong + 1
        ),
        rounds(K + 1, Last, Doubler, Summer, Host, !Wrong, !IO)
    ).

:- func expected_sum(int, int, int) = int.
expected_sum(K, I, S) =
    ( if I < K then expected_sum(K, I + 1, S + (K + I) mod 256) else S ).

% CHECK:0 wrong of 200
% CHECK-NEXT:42 a string after the arrays
