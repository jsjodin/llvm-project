% DESIGN: GPU processes / The device runtime
% A GPU process makes, reads, sets, fills and resizes arrays of flat
% records in place, and sends and receives them, alone or as a message's
% last field, to the host and to another GPU process, copied as one block.
% The records here have fields of 1, 4 and 8 bytes, so C's layout pads
% them. A mover answers each array with its cells moved, in reverse order;
% a filler answers a batch with its cells, then its fill twice, after a
% fill and two resizes, each warmed by its index; a forwarder moves each
% array and passes it to a summer, another GPU process, which reports its
% totals. Run with no argument the kinds run at the host, which the oracle
% checks; run-gpu-record-arrays-gpu.test runs them on the GPU and checks
% the same output.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_record_arrays_translation.erl
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:message(array([cell(point(109i32, -110i32), 108u8, red, 3.0), cell(point(71i32, -73i32), 71u8, blue, 1.5), cell(point(33i32, -36i32), 34u8, green, 0.0)]))
% OUTPUT-NEXT:message(array([cell(point(11i32, -12i32), 12u8, green, 0.0)]))
% OUTPUT-NEXT:message(array([]))
% OUTPUT-NEXT:message(array([cell(point(0i32, -2i32), 22u8, red, 0.0), cell(point(1i32, -2i32), 59u8, green, 3.0), cell(point(4i32, -9i32), 247u8, green, 38.0), cell(point(4i32, -9i32), 247u8, green, 39.0)]))
% OUTPUT-NEXT:message(array([cell(point(0i32, 0i32), 0u8, red, 0.0), cell(point(0i32, 0i32), 0u8, red, 1.0)]))
% OUTPUT-NEXT:message(totals(4, -10, 12))
% OUTPUT-NEXT:message(totals(1000, -500500, 249750000))
% OUTPUT-NOT:{{.}}
:- module run_gpu_record_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, process, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type point
    --->    point(x :: int32, y :: int32).

:- type cell
    --->    cell(
                at :: point,
                count :: uint8,
                hue :: colour,
                heat :: float
            ).

:- type batch
    --->    batch(fill :: cell, cells :: array(cell)).

:- type totals
    --->    totals(int, int, int).    % cells, sum of x + y, sum of heats

    % Answers each array until an empty one, which it answers too.
:- process mover(mailbox(array(cell))) receives array(cell) on gpu.

mover(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(A),
        N = array.size(A),
        send(Reply, moved(A, 0, array.init(N, still)), !IO),
        ( if N > 0 then
            mover(Reply, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

    % Answers each batch until one of no cells, which it answers too.
:- process filler(mailbox(array(cell))) receives batch on gpu.

filler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(batch(Fill, A)),
        N = array.size(A),
        send(Reply, filled(Fill, A), !IO),
        ( if N > 0 then
            filler(Reply, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

    % Moves each array and passes it on, until an empty one, which it
    % passes on too.
:- process forwarder(mailbox(array(cell))) receives array(cell) on gpu.

forwarder(Next, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(A),
        N = array.size(A),
        send(Next, moved(A, 0, array.init(N, still)), !IO),
        ( if N > 0 then
            forwarder(Next, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

    % Reports the totals of each array, until an empty one.
:- process summer(mailbox(totals)) receives array(cell) on gpu.

summer(Report, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(A),
        ( if array.size(A) > 0 then
            send(Report, sum_cells(A, 0, totals(0, 0, 0)), !IO),
            summer(Report, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

:- func sum_cells(array(cell), int, totals) = totals.

sum_cells(A, I, Totals0) = Totals :-
    ( if I < array.size(A) then
        C = array.lookup(A, I),
        Totals0 = totals(N, S, H),
        Totals = sum_cells(A, I + 1, totals(N + 1,
            S + int32.to_int(C ^ at ^ x) + int32.to_int(C ^ at ^ y),
            H + float.truncate_to_int(C ^ heat)))
    else
        Totals = Totals0
    ).

    % The cells, then Fill twice, each warmed by its index.
:- func filled(cell, array(cell)) = array(cell).

filled(Fill, A) = B :-
    N = array.size(A),
    array.init(N, still, B0),
    copy_cells(0, A, B0, B1),
    array.resize(N + 1, Fill, B1, B2),
    array.fill(Fill, B2, B3),
    copy_cells(0, A, B3, B4),
    array.resize(N + 2, Fill, B4, B5),
    warm(0, B5, B).

:- pred copy_cells(int::in, array(cell)::in,
    array(cell)::array_di, array(cell)::array_uo) is det.

copy_cells(I, A, !B) :-
    ( if I < array.size(A) then
        array.lookup(A, I, C),
        array.set(I, C, !B),
        copy_cells(I + 1, A, !B)
    else
        true
    ).

:- pred warm(int::in, array(cell)::array_di, array(cell)::array_uo) is det.

warm(I, !B) :-
    ( if I < array.size(!.B) then
        array.lookup(!.B, I, C),
        array.set(I, C ^ heat := C ^ heat + float(I), !B),
        warm(I + 1, !B)
    else
        true
    ).

:- func still = cell.

still = cell(point(0i32, 0i32), 0u8, red, 0.0).

    % Each cell moved by its count, its count plus one, its colour the next
    % and its heat halved, in reverse order.
:- func moved(array(cell), int, array(cell)) = array(cell).
:- mode moved(in, in, array_di) = array_uo is det.

moved(A, I, B) =
    ( if I < array.size(A) then
        moved(A, I + 1, B ^ elem(array.size(A) - 1 - I) := move(A ^ elem(I)))
    else
        B
    ).

:- func move(cell) = cell.

move(cell(point(X, Y), C, H, T)) = cell(point(X + N, Y - N), C + 1u8,
        next(H), T / 2.0) :-
    N = int32.cast_from_int(uint8.to_int(C)).

:- func next(colour) = colour.

next(red) = green.
next(green) = blue.
next(blue) = red.

    % Cells 0 .. N - 1 of round N.
:- func round(int) = array(cell).

round(N) = array.generate(N, element(N)).

:- func element(int, int) = cell.

element(N, I) = cell(point(int32.cast_from_int(I), int32.cast_from_int(-N)),
    uint8.cast_from_int((I * 37 + N * 11) mod 256),
    ( if I mod 3 = 0 then red else if I mod 3 = 1 then green else blue ),
    float(I * N)).

:- pred answer(mailbox(T)::in, io::di, io::uo) is det.

answer(Mailbox, !IO) :-
    receive(Mailbox, any, infinity, R, !IO),
    io.write_line(R, !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Arrays, !IO),
    new_mailbox(Totals, !IO),
    spawn(Place, mover(Arrays), M, !IO),
    list.foldl(send_round(M, Arrays), [3, 1, 0], !IO),
    spawn(Place, filler(Arrays), F, !IO),
    send(F, batch(element(9, 4), round(2)), !IO),
    answer(Arrays, !IO),
    send(F, batch(still, round(0)), !IO),
    answer(Arrays, !IO),
    spawn(Place, summer(Totals), S, !IO),
    spawn(Place, forwarder(S), W, !IO),
    list.foldl(send_to_sum(W, Totals), [4, 1000], !IO),
    send(W, round(0), !IO).

:- pred send_round(mailbox(array(cell))::in, mailbox(array(cell))::in,
    int::in, io::di, io::uo) is det.

send_round(Mover, Arrays, N, !IO) :-
    send(Mover, round(N), !IO),
    answer(Arrays, !IO).

:- pred send_to_sum(mailbox(array(cell))::in, mailbox(totals)::in,
    int::in, io::di, io::uo) is det.

send_to_sum(Forwarder, Totals, N, !IO) :-
    send(Forwarder, round(N), !IO),
    answer(Totals, !IO).
