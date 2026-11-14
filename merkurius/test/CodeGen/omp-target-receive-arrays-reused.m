% Target regions, one after another, receive arrays of 1024 to 65536
% words, each element a function of the region, the array and its index,
% and count the elements that are not. Each region first fills an array of
% its own, so that the arena's memory, which the next region allocates
% again, has been written by the device. On a GPU (the -gpu test) arrays of
% 8 KiB and more on gfx9 are copied by the host's copy engine straight into
% the arena (runtime/offload.cpp, rpcTakeArrayInto), which must not leave
% the device reading what its caches held of that memory before.
%
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential && %t.sequential | FileCheck %s --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_receive_arrays_reused.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- func regions = int.
regions = 40.

:- func per_region = int.
per_region = 12.

:- func words(int) = int.
words(J) = 1024 << (J mod 7).

:- func value(int, int, int) = int.
value(R, J, I) = R * 1000003 + J * 7919 + I.

:- pred fill(int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
fill(R, J, I, !A) :-
    ( if I < array.size(!.A) then
        array.set(I, value(R, J, I), !A),
        fill(R, J, I + 1, !A)
    else
        true
    ).

:- func wrong(array(int), int, int, int, int) = int.
wrong(A, R, J, I, W) =
    ( if I >= array.size(A) then
        W
    else if A ^ elem(I) = value(R, J, I) then
        wrong(A, R, J, I + 1, W)
    else
        wrong(A, R, J, I + 1, W + 1)
    ).

:- pred take(mailbox(array(int))::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
take(In, R, J, W0, W, !IO) :-
    ( if J < per_region then
        receive(In, any, after(60000), Got, !IO),
        (
            Got = message(A),
            ( if array.size(A) = words(J) then
                W1 = wrong(A, R, J, 0, W0)
            else
                W1 = W0 + array.size(A) + 1
            ),
            take(In, R, J + 1, W1, W, !IO)
        ;
            Got = timeout,
            W = W0 + 1000000
        )
    else
        W = W0
    ).

:- pred send_region(mailbox(array(int))::in, int::in, int::in,
    io::di, io::uo) is det.
send_region(In, R, J, !IO) :-
    ( if J < per_region then
        fill(R, J, 0, array.init(words(J), 0), A),
        send(In, A, !IO),
        send_region(In, R, J + 1, !IO)
    else
        true
    ).

:- pred regions(mailbox(array(int))::in, mailbox(int)::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
regions(In, Out, R, W0, W, !IO) :-
    ( if R < regions then
        send_region(In, R, 0, !IO),
        %$omp target
        (
            fill(R, -1, 0, array.init(524288, 0), Scratch),
            take(In, R, 0, 0, Wrong, !IO),
            send(Out, Wrong + (Scratch ^ elem(0)) * 0, !IO)
        ),
        receive(Out, any, after(60000), Got, !IO),
        ( Got = message(N), W1 = W0 + N ; Got = timeout, W1 = W0 + 1000000 ),
        regions(In, Out, R + 1, W1, W, !IO)
    else
        W = W0
    ).

main(!IO) :-
    new_mailbox(In : mailbox(array(int)), !IO),
    new_mailbox(Out : mailbox(int), !IO),
    regions(In, Out, 0, 0, Wrong, !IO),
    io.write_int(Wrong, !IO),
    io.write_string(" wrong of ", !IO),
    io.write_int(regions * per_region, !IO),
    io.write_string(" arrays\n", !IO).

% CHECK:0 wrong of 480 arrays
