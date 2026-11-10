% A polymorphic procedure that a target region calls cannot send on the
% device yet: its message type's key is not known at compile time, and the
% device has no type_infos to send instead (docs/INPROGRESS.md, The
% device's side of the first slice, step 4). Its send stays the call of
% process.send, which the device's compile copies from the library, and
% whose foreign code it does not compile yet: an error, with a note for each
% call back to the region's code.
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
% CHECK: process.m:{{[0-9]+}}:{{[0-9]+}}: error: cannot call process.send_message_8_0 on an OpenMP device yet: its foreign code calls merk_mailbox_send, which is not on the device
% CHECK-NEXT: process.m:{{[0-9]+}}:{{[0-9]+}}: note: called by process.send_4_0 here: process.send_message_8_0
% CHECK-NEXT: omp-target-send-polymorphic.m:[[@LINE-3]]:5: note: called by omp_target_send_polymorphic.forward_4_0 here: process.send_4_0

main(!IO) :-
    new_mailbox(Mailbox, !IO),
    %$omp target
    forward(Mailbox, 7, !IO),
    receive(Mailbox, any, after(0), Got, !IO),
    io.print_line(Got, !IO).

% OUTPUT:message(7)
