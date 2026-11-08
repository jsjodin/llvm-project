% A target region receives arrays of one-word elements: small ones, an
% empty one and one of 2 MiB, then one that it copies, updates and sends
% back, in a procedure it calls; then a timeout passes. Each is a call of
% merk_device_receive_array. On a GPU (the -gpu test) a poll gives the
% oldest message's size, the device allocates the array in its arena, and
% the host's callback copies the message there, given its address.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_receive_array.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, maybe, process.

:- func total(array(int), int, int) = int.
total(A, I, S) =
    ( if I >= array.size(A) then S else total(A, I + 1, S + A ^ elem(I)) ).

:- pred take(mailbox(array(int))::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
take(In, N, S0, S, !IO) :-
    ( if N > 0 then
        receive(In, any, infinity, R, !IO),
        (
            R = yes(A),
            take(In, N - 1, S0 + total(A, 0, 0), S, !IO)
        ;
            R = no,
            S = -1
        )
    else
        S = S0
    ).

:- pred echo(mailbox(array(int))::in, mailbox(array(int))::in,
    io::di, io::uo) is det.
echo(In, Back, !IO) :-
    receive(In, any, after(1000), R, !IO),
    (
        R = yes(A),
        array.copy(A, C0),
        array.set(0, 99, C0, C),
        send(Back, C, !IO)
    ;
        R = no
    ).

:- pred count_up(int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
count_up(I, N, !A) :-
    ( if I < N then
        array.set(I, I, !A),
        count_up(I + 1, N, !A)
    else
        true
    ).

main(!IO) :-
    new_mailbox(In : mailbox(array(int)), !IO),
    new_mailbox(Out : mailbox(int), !IO),
    new_mailbox(Back : mailbox(array(int)), !IO),
    send(In, array.from_list([1, 2, 3]), !IO),
    send(In, array.init(0, 0), !IO),
    count_up(0, 262144, array.init(262144, 0), Big),
    send(In, Big, !IO),
    send(In, array.from_list([10, 20]), !IO),
    %$omp target
    (
        take(In, 3, 0, Sum, !IO),
        send(Out, Sum, !IO),
        echo(In, Back, !IO),
        receive(In, any, after(20), Late, !IO),
        (
            Late = yes(_),
            send(Out, -1, !IO)
        ;
            Late = no,
            send(Out, -2, !IO)
        )
    ),
% MERK-LABEL: merk.proc @omp_target_receive_array.main_2_0()
% MERK: omp.target
% MERK: merk.foreign_call @merk_device_receive_array(
    receive(Out, any, after(0), R1, !IO),
    receive(Out, any, after(0), R2, !IO),
    io.print_line({R1, R2}, !IO),
    receive(Back, any, after(0), B, !IO),
    (
        B = yes(C),
        io.write_int(array.size(C), !IO), io.write_string(" ", !IO),
        io.write_int(C ^ elem(0), !IO), io.write_string(" ", !IO),
        io.write_int(C ^ elem(1), !IO), io.nl(!IO)
    ;
        B = no,
        io.write_string("nothing back\n", !IO)
    ).

% OUTPUT:{yes(34359607302), yes(-2)}
% OUTPUT-NEXT:2 99 20
