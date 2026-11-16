% Arrays of flat records between the host and process kinds that run on a
% GPU (docs/INPROGRESS.md, Flat records, step 2b). A mover takes arrays of 0
% to 40 cells, which hold them inline (runtime/array.h), and answers each
% with an array it makes, each cell moved, in reverse order. A filler takes
% `batch(Fill, Cells)`, a fixed part followed by an array of records, and
% answers with an array it makes of Cells, then Fill twice, set, filled and
% resized in place, and each cell warmed where it is. Messages are copied
% by their type_infos (runtime/copy.h), which copy an array of records as
% one block, its header's two words and its records' bytes. The kinds run
% at the host here, and on the GPU in process-record-arrays-gpu.test.
% The -O0 build has no --offload-arch, as process-records.m's.
%
% mmc has no process module to run it with, so the output is written by
% hand: each round's answer is checked against what the host computes.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t host 2>/dev/null | FileCheck %s --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t2 host 2>/dev/null | FileCheck %s --match-full-lines --strict-whitespace

:- module process_record_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, int32, list, process, string, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type point ---> point(x :: int32, y :: int32).
:- type cell
    --->    cell(
                at :: point,
                count :: uint8,
                hue :: colour,
                heat :: float
            ).

    % Answers each array until an empty one, which it answers too.
:- process mover(mailbox(array(cell))) receives array(cell) on gpu.

mover(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(A) then
        N = array.size(A),
        send(Reply, moved(A, 0, array.init(N, still)), !IO),
        ( if N > 0 then
            mover(Reply, Self, !IO)
        else
            true
        )
    else
        true
    ).

:- type batch ---> batch(fill :: cell, cells :: array(cell)).

    % Answers each batch until one of no cells, which it answers too.
:- process filler(mailbox(array(cell))) receives batch on gpu.

filler(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(batch(Fill, A)) then
        N = array.size(A),
        send(Reply, filled(Fill, A), !IO),
        ( if N > 0 then
            filler(Reply, Self, !IO)
        else
            true
        )
    else
        true
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

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reply, !IO),
    spawn(Place, mover(Reply), M, !IO),
    rounds(M, Reply, 1, 40, 0, Wrong, !IO),
    send(M, round(0), !IO),
    receive(Reply, any, after(10000), Last, !IO),
    ( if Last = message(L), array.size(L) = 0 then
        io.write_string("empty answered\n", !IO)
    else
        io.write_string("empty not answered\n", !IO)
    ),
    io.write_int(Wrong, !IO),
    io.write_string(" wrong of 40\n", !IO),
    spawn(Place, filler(Reply), F, !IO),
    batches(F, Reply, 1, 40, 0, Wrong2, !IO),
    send(F, batch(still, round(0)), !IO),
    receive(Reply, any, after(10000), Last2, !IO),
    ( if Last2 = message(L2) then
        io.print_line(L2, !IO)
    else
        io.write_string("no batch answered\n", !IO)
    ),
    io.write_int(Wrong2, !IO),
    io.write_string(" batches wrong of 40\n", !IO).

    % Sends batches N .. Max and checks the answers.
:- pred batches(mailbox(batch)::in, mailbox(array(cell))::in, int::in,
    int::in, int::in, int::out, io::di, io::uo) is det.

batches(F, Reply, N, Max, !Wrong, !IO) :-
    ( if N =< Max then
        A = round(N),
        Fill = element(N + 1, N),
        send(F, batch(Fill, A), !IO),
        receive(Reply, any, after(10000), R, !IO),
        ( if
            R = message(B),
            B = filled(Fill, A)
        then
            true
        else
            !:Wrong = !.Wrong + 1,
            io.format("batch %d: ", [i(N)], !IO),
            io.print_line(R, !IO)
        ),
        batches(F, Reply, N + 1, Max, !Wrong, !IO)
    else
        true
    ).

    % Sends rounds N .. Max and checks the answers.
:- pred rounds(mailbox(array(cell))::in, mailbox(array(cell))::in, int::in,
    int::in, int::in, int::out, io::di, io::uo) is det.

rounds(M, Reply, N, Max, !Wrong, !IO) :-
    ( if N =< Max then
        A = round(N),
        send(M, A, !IO),
        receive(Reply, any, after(10000), R, !IO),
        ( if
            R = message(B),
            B = moved(A, 0, array.init(N, still))
        then
            true
        else
            !:Wrong = !.Wrong + 1,
            io.format("round %d: ", [i(N)], !IO),
            io.print_line(R, !IO)
        ),
        rounds(M, Reply, N + 1, Max, !Wrong, !IO)
    else
        true
    ).

% CHECK:empty answered
% CHECK-NEXT:0 wrong of 40
% CHECK-NEXT:array([cell(point(0i32, 0i32), 0u8, red, 0.0), cell(point(0i32, 0i32), 0u8, red, 1.0)])
% CHECK-NEXT:0 batches wrong of 40
