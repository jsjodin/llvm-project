% A target region sends and receives messages of any type, which are copied
% by their type_info (docs/INPROGRESS.md, GPU processes without omp
% target, step 4e): here a point, a string, a list and an array of strings,
% each of which the region receives from the host and sends back changed
% or as it is. On a GPU (the -gpu test) the host copies a message for the
% region into host memory that the GPU copies into its arena, and the GPU
% copies one it sends into host memory that the host queues from.
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
:- import_module array, int, list, process.

:- type msg
    --->    point(int, int)
    ;       label(string)
    ;       numbers(list(int))
    ;       words(array(string)).

    % Receives N messages from In and sends each to Out: a point with its
    % coordinates swapped, numbers with their sum in front, and the others
    % as they are.
:- pred relay(mailbox(msg)::in, mailbox(msg)::in, int::in,
    io::di, io::uo) is det.

relay(In, Out, N, !IO) :-
    ( if N > 0 then
        receive(In, any, after(10000), R, !IO),
        ( if R = message(M) then
            send(Out, changed(M), !IO)
        else
            send(Out, label("no message"), !IO)
        ),
        relay(In, Out, N - 1, !IO)
    else
        true
    ).

:- func changed(msg) = msg.

changed(point(X, Y)) = point(Y, X).
changed(label(S)) = label(S).
changed(numbers(Ns)) = numbers([sum(Ns) | Ns]).
changed(words(Ws)) = words(Ws).

:- func sum(list(int)) = int.

sum([]) = 0.
sum([N | Ns]) = N + sum(Ns).

% DEVICE: call i32 @merk_device_receive_term(ptr %{{.*}}, i64 %{{.*}}, ptr @"merk.type_ctor.omp_target_term_messages.msg/0", ptr %{{.*}})
% DEVICE: call void @merk_device_send_term(ptr %{{.*}}, i64 {{-?[0-9]+}}, i64 %{{.*}}, ptr @"merk.type_ctor.omp_target_term_messages.msg/0")

main(!IO) :-
    new_mailbox(In, !IO),
    new_mailbox(Out, !IO),
    send(In, point(1, 2), !IO),
    send(In, label("a label"), !IO),
    send(In, numbers([1, 2, 3, 4]), !IO),
    send(In, words(array(["one", "two"])), !IO),
    %$omp target
    relay(In, Out, 4, !IO),
    print_all(Out, 4, !IO).

:- pred print_all(mailbox(msg)::in, int::in, io::di, io::uo) is det.

print_all(Out, N, !IO) :-
    ( if N > 0 then
        receive(Out, any, after(0), R, !IO),
        io.print_line(R, !IO),
        print_all(Out, N - 1, !IO)
    else
        true
    ).

% OUTPUT:message(point(2, 1))
% OUTPUT-NEXT:message(label("a label"))
% OUTPUT-NEXT:message(numbers([10, 1, 2, 3, 4]))
% OUTPUT-NEXT:message(words(array(["one", "two"])))
