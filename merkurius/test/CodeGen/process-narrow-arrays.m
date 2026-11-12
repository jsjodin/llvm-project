% Narrow arrays between the host and a process kind that runs on a GPU
% (runtime/array.h). A mirror takes uint8 arrays of 0 to 70 elements, which
% cross the words their bytes share, and answers each with three arrays it
% makes: the elements reversed and each plus one, as uint8s; each minus
% 128, as int16s; and each's colour, by its value. The ring shows the device
% each array's header, from which it copies the array's words; the device
% copies its answers into host memory by theirs. The program runs its kind
% at the host here, where messages are copied by their type_infos
% (runtime/copy.h), and at the GPU in process-narrow-arrays-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand: each round's answers are checked against what the host computes.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_GC_EVERY_ALLOC=1 MERK_CHECK_ROOTS=1 %t host | FileCheck %s --match-full-lines --strict-whitespace

:- module process_narrow_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, int16, list, process, string, uint8.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type replies
    --->    replies(
                mailbox(array(uint8)),
                mailbox(array(int16)),
                mailbox(array(colour))
            ).

    % Answers each array until an empty one, which it answers too.
:- process mirror(mailbox(array(uint8)), mailbox(array(int16)),
    mailbox(array(colour))) receives array(uint8) on gpu.

mirror(Bytes, Shorts, Colours, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(A) then
        N = array.size(A),
        send(Bytes, reversed_plus_one(A, N, 0, array.init(N, 0u8)), !IO),
        send(Shorts, shifted(A, 0, array.init(N, 0i16)), !IO),
        send(Colours, coloured(A, 0, array.init(N, red)), !IO),
        ( if N > 0 then
            mirror(Bytes, Shorts, Colours, Self, !IO)
        else
            true
        )
    else
        true
    ).

:- func reversed_plus_one(array(uint8), int, int, array(uint8)) = array(uint8).
:- mode reversed_plus_one(in, in, in, array_di) = array_uo is det.
reversed_plus_one(A, N, I, B) =
    ( if I < N then
        reversed_plus_one(A, N, I + 1,
            B ^ elem(N - 1 - I) := A ^ elem(I) + 1u8)
    else
        B
    ).

:- func shifted(array(uint8), int, array(int16)) = array(int16).
:- mode shifted(in, in, array_di) = array_uo is det.
shifted(A, I, B) =
    ( if I < array.size(A) then
        shifted(A, I + 1, B ^ elem(I) :=
            int16.cast_from_int(uint8.to_int(A ^ elem(I)) - 128))
    else
        B
    ).

:- func coloured(array(uint8), int, array(colour)) = array(colour).
:- mode coloured(in, in, array_di) = array_uo is det.
coloured(A, I, B) =
    ( if I < array.size(A) then
        coloured(A, I + 1, B ^ elem(I) := colour_of(A ^ elem(I)))
    else
        B
    ).

:- func colour_of(uint8) = colour.
colour_of(X) = ( if X < 85u8 then red else if X < 170u8 then green else blue ).

    % Elements 0 .. N - 1 of round N, spread over 0 to 255.
:- func round(int) = array(uint8).
round(N) = array.generate(N, element(N)).

:- func element(int, int) = uint8.
element(N, I) = uint8.cast_from_int((I * 37 + N * 11) mod 256).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Bytes, !IO),
    new_mailbox(Shorts, !IO),
    new_mailbox(Colours, !IO),
    spawn(Place, mirror(Bytes, Shorts, Colours), M, !IO),
    rounds(M, replies(Bytes, Shorts, Colours), 1, 70, 0, Wrong, !IO),
    send(M, round(0), !IO),
    receive(Bytes, any, after(10000), Last, !IO),
    ( if Last = message(L), array.size(L) = 0 then
        io.write_string("empty answered\n", !IO)
    else
        io.write_string("empty not answered\n", !IO)
    ),
    io.write_int(Wrong, !IO),
    io.write_string(" wrong of 70\n", !IO),
    receive(Shorts, any, after(10000), _, !IO),
    receive(Colours, any, after(10000), _, !IO).

    % Sends rounds N .. Max and checks the answers.
:- pred rounds(mailbox(array(uint8))::in, replies::in, int::in, int::in,
    int::in, int::out, io::di, io::uo) is det.

rounds(M, Replies, N, Max, !Wrong, !IO) :-
    ( if N =< Max then
        A = round(N),
        send(M, A, !IO),
        Replies = replies(Bytes, Shorts, Colours),
        receive(Bytes, any, after(10000), RB, !IO),
        receive(Shorts, any, after(10000), RS, !IO),
        receive(Colours, any, after(10000), RC, !IO),
        ( if
            RB = message(B),
            B = reversed_plus_one(A, N, 0, array.init(N, 0u8)),
            RS = message(S),
            S = shifted(A, 0, array.init(N, 0i16)),
            RC = message(C),
            C = coloured(A, 0, array.init(N, red))
        then
            true
        else
            !:Wrong = !.Wrong + 1,
            io.format("round %d: ", [i(N)], !IO),
            io.print(RB, !IO), io.write_string(" ", !IO),
            io.print(RS, !IO), io.write_string(" ", !IO),
            io.print_line(RC, !IO)
        ),
        rounds(M, Replies, N + 1, Max, !Wrong, !IO)
    else
        true
    ).

% CHECK:empty answered
% CHECK-NEXT:0 wrong of 70
