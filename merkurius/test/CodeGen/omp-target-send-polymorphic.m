% A polymorphic procedure that a target region calls cannot send on the
% device yet: its message type's key is not known at compile time, and the
% device has no type_infos to send instead (docs/INPROGRESS.md, The
% device's side of the first slice, step 4). Its send stays the call of
% process.send, an error in the device's compile since a region calls it.
% On the host it runs.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: amdgpu-registered-target, libomp

:- module omp_target_send_polymorphic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, maybe, process.

:- pred forward(mailbox(T)::in, T::in, io::di, io::uo) is det.
forward(Mailbox, Message, !IO) :-
    send(Mailbox, Message, !IO).
% CHECK: omp-target-send-polymorphic.m:[[@LINE-1]]:5: error: cannot call process.send_4_0 on an OpenMP device yet: it is not compiled for the device

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    forward(Mailbox, 7, !IO),
    receive(Mailbox, any, after(0), Got, !IO),
    io.print_line(Got, !IO).

% OUTPUT:message(7)
