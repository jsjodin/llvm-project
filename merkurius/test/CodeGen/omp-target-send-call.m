% A send in a procedure that a target region calls, rather than in the
% region itself, is an error in the device's compile: the type_info that the
% host encodes the message by is made before the region, and a procedure
% has no region before it. Sends in such procedures are to come with the
% first slice's loop (docs/INPROGRESS.md, The device's side of the first
% slice). On the host it runs.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_send_call.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pred twice(mailbox(int)::in, int::in, io::di, io::uo) is det.
twice(Mailbox, N, !IO) :-
    send(Mailbox, N, !IO),
% CHECK: omp-target-send-call.m:[[@LINE-1]]:5: error: cannot call process.send_4_0 on an OpenMP device yet: it is not compiled for the device
    send(Mailbox, N, !IO).

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    twice(Mailbox, 5, !IO),
    receive(Mailbox, any, after(0), A, !IO),
    receive(Mailbox, any, after(0), B, !IO),
    io.print_line({A, B}, !IO).

% OUTPUT:{yes(5), yes(5)}
