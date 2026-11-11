% A target region sends and receives flat messages (design document,
% "Messages and memory on the device"): here a record, a record of
% records, a constructor of an enumeration, a character and a float, and a
% string, each of which the region receives from the host and sends back
% changed or as it is. A message that is not flat is an error
% (omp-target-send-errors.m). On a GPU (the -gpu test) the host copies a
% message for the region into host memory that the GPU copies into its
% arena, and the GPU copies one it sends into host memory that the host
% queues from.
% Temporary (docs/INPROGRESS.md, GPU processes without omp target, step 5,
% "Flat records"): the copies walk the messages' type_info, until flat
% records are values on the device and travel as their bytes.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% RUN: merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential.O2 && %t.sequential.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_term_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, process.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type pos
    --->    pos(int, int).

:- type msg
    --->    point(int, int)
    ;       segment(pos, pos)
    ;       tagged(character, colour, float).

    % Receives N messages from In and sends each to Out: a point with its
    % coordinates swapped, a segment with its ends swapped, and the others
    % as they are; then a string from Words, which it sends back.
:- pred relay(mailbox(msg)::in, mailbox(msg)::in, mailbox(string)::in,
    int::in, io::di, io::uo) is det.

relay(In, Out, Words, N, !IO) :-
    ( if N > 0 then
        receive(In, any, after(10000), R, !IO),
        ( if R = message(M) then
            send(Out, changed(M), !IO)
        else
            send(Out, point(-1, -1), !IO)
        ),
        relay(In, Out, Words, N - 1, !IO)
    else
        receive(Words, any, after(10000), W, !IO),
        ( if W = message(S) then
            send(Words, S, !IO)
        else
            send(Words, "no message", !IO)
        )
    ).

:- func changed(msg) = msg.

changed(point(X, Y)) = point(Y, X).
changed(segment(A, B)) = segment(B, A).
changed(tagged(C, Colour, F)) = tagged(C, Colour, F).

main(!IO) :-
    new_mailbox(In, !IO),
    new_mailbox(Out, !IO),
    new_mailbox(Words, !IO),
    send(In, point(1, 2), !IO),
    send(In, segment(pos(1, 2), pos(3, 4)), !IO),
    send(In, tagged('x', blue, 2.5), !IO),
    send(Words, "a string", !IO),
    %$omp target
    relay(In, Out, Words, 3, !IO),
    print_all(Out, 3, !IO),
    receive(Words, any, after(0), W, !IO),
    io.print_line(W, !IO).

:- pred print_all(mailbox(msg)::in, int::in, io::di, io::uo) is det.

print_all(Out, N, !IO) :-
    ( if N > 0 then
        receive(Out, any, after(0), R, !IO),
        io.print_line(R, !IO),
        print_all(Out, N - 1, !IO)
    else
        true
    ).

% DEVICE: call i32 @merk_device_receive_term(ptr %{{.*}}, i64 %{{.*}}, ptr @"merk.type_ctor.omp_target_term_messages.msg/0", ptr %{{.*}})
% DEVICE: call void @merk_device_send_term(ptr %{{.*}}, i64 {{-?[0-9]+}}, i64 %{{.*}}, ptr @"merk.type_ctor.omp_target_term_messages.msg/0")

% OUTPUT:message(point(2, 1))
% OUTPUT-NEXT:message(segment(pos(3, 4), pos(1, 2)))
% OUTPUT-NEXT:message(tagged('x', blue, 2.5))
% OUTPUT-NEXT:message("a string")
