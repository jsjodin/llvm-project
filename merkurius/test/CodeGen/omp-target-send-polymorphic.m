% A polymorphic procedure that a target region calls cannot send on the
% device yet: its message type's key is not known at compile time, and the
% device has no type_infos to send instead (docs/INPROGRESS.md, The
% device's side of the first slice, step 4). Its merk.send says so, and
% the device's compile reports it where the device would run it.
% On the host it runs.
%
% RUN: not merkc -fc1 -fopenmp -fopenmp-is-target-device -triple amdgpu10.30-amd-amdhsa -emit-mlir %s 2>&1 | FileCheck %s --implicit-check-not=error:
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
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
% CHECK: omp-target-send-polymorphic.m:[[@LINE-1]]:5: error: 'merk.send' op sends messages of a type that a device does not send, which cannot be done on an OpenMP device yet

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    forward(Mailbox, 7, !IO),
    receive(Mailbox, any, after(0), Got, !IO),
    io.print_line(Got, !IO).

% OUTPUT:message(7)
