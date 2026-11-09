% A target region sends arrays of one-word elements: arrays it made,
% an empty one, a large one of 2 MiB, and one it was given, some in a
% procedure it calls. Each is a call of merk_device_send_array, by key as
% other messages; on a GPU (the -gpu test) the host's callback copies the
% array from the device, given its address and size, and queues it.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fc1 -fopenmp -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_send_array.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, maybe, process.

:- pred count_up(int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
count_up(I, N, !A) :-
    ( if I < N then
        array.set(I, I, !A),
        count_up(I + 1, N, !A)
    else
        true
    ).

:- pred send_counted(mailbox(array(int))::in, int::in,
    io::di, io::uo) is det.
send_counted(Ints, N, !IO) :-
    A0 = array.init(N, 0),
    count_up(0, N, A0, A),
    send(Ints, A, !IO).

main(!IO) :-
    new_mailbox(Ints : mailbox(array(int)), !IO),
    new_mailbox(Floats : mailbox(array(float)), !IO),
    Given = array.from_list([4, 5, 6]) : array(int),
    %$omp target
    (
        send(Ints, array.init(3, 7), !IO),
        send(Ints, array.init(0, 0), !IO),
        send_counted(Ints, 262144, !IO),
        send(Ints, Given, !IO),
        send(Floats, array.init(2, 1.5), !IO)
    ),
% MERK-LABEL: merk.proc @omp_target_send_array.main_2_0()
% MERK: omp.target
% MERK: merk.foreign_call @merk_device_send_array(
    receive_ints(Ints, !IO),
    receive(Floats, any, after(0), F, !IO),
    (
        F = message(A),
        io.write_int(array.size(A), !IO), io.write_string(" ", !IO),
        io.write_float(A ^ elem(1), !IO), io.nl(!IO)
    ;
        F = timeout,
        io.write_string("no floats\n", !IO)
    ).

:- func total(array(int), int, int) = int.
total(A, I, S) =
    ( if I >= array.size(A) then S else total(A, I + 1, S + A ^ elem(I)) ).

:- pred receive_ints(mailbox(array(int))::in, io::di, io::uo) is det.
receive_ints(Ints, !IO) :-
    receive(Ints, any, after(0), R, !IO),
    (
        R = message(A),
        io.write_int(array.size(A), !IO), io.write_string(" ", !IO),
        io.write_int(total(A, 0, 0), !IO), io.nl(!IO),
        receive_ints(Ints, !IO)
    ;
        R = timeout,
        io.write_string("end\n", !IO)
    ).

% OUTPUT:3 21
% OUTPUT-NEXT:0 0
% OUTPUT-NEXT:262144 34359607296
% OUTPUT-NEXT:3 15
% OUTPUT-NEXT:end
% OUTPUT-NEXT:2 1.5
