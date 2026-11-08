% A target region sends only one-word messages and arrays of them so far:
% integers, floats, characters and constants of a type of constants only.
% A message of another type, such as a string, a term with arguments or an
% array of strings, is an error at
% the send, in the host's compile as in the device's, since both compile
% the region (docs/INPROGRESS.md, The device's side of the first slice,
% Decision 3).
%
% RUN: not merkc -fc1 -fopenmp -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-merk %s 2>&1 | FileCheck %s --implicit-check-not=error:
% REQUIRES: amdgpu-registered-target

:- module omp_target_send_errors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, process.

:- type point
    --->    point(int, int).

:- pred greet(mailbox(string)::in, io::di, io::uo) is det.
greet(Strings, !IO) :-
    %$omp target
    send(Strings, "hello", !IO).
% CHECK: omp-target-send-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for sending messages of type string in target regions

:- pred words(mailbox(array(string))::in, io::di, io::uo) is det.
words(Words, !IO) :-
    %$omp target
    send(Words, array.init(1, "a"), !IO).
% CHECK: omp-target-send-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for sending messages of type array.array(string) in target regions

:- pred origin(mailbox(point)::in, io::di, io::uo) is det.
origin(Points, !IO) :-
    %$omp target
    send(Points, point(0, 0), !IO).
% CHECK: omp-target-send-errors.m:[[@LINE-1]]:5: error: not yet implemented: code generation for sending messages of type omp_target_send_errors.point in target regions

main(!IO) :-
    new_mailbox(Strings, !IO),
    greet(Strings, !IO),
    new_mailbox(Words, !IO),
    words(Words, !IO),
    new_mailbox(Points, !IO),
    origin(Points, !IO).
