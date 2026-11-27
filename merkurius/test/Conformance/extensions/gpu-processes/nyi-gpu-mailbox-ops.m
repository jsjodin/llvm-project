% DESIGN: GPU processes / The device runtime
% A GPU process cannot make, close or stop a mailbox yet: the device
% runtime has none of these. Each is reported at its operation
% (docs/INPROGRESS.md, Concurrency in merk, step 2 (b)). A monitor is
% rejected before, as its monitor_ref and function are not flat values.
% ORACLE: none
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: not merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:
:- module nyi_gpu_mailbox_ops.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.

:- process manager(mailbox(int)) receives int on gpu.

:- implementation.

manager(Other, Self, !IO) :-
    new_mailbox(Mine : mailbox(int), !IO),
% CHECK: nyi-gpu-mailbox-ops.m:[[@LINE-1]]:5: error: 'merk.new_mailbox' op makes a mailbox, which cannot be done in a GPU process yet
    send(Other, 1, !IO),
    stop(Other, !IO),
% CHECK: nyi-gpu-mailbox-ops.m:[[@LINE-1]]:5: error: 'merk.stop' op stops a mailbox's owner, which cannot be done in a GPU process yet
    close_mailbox(Self, !IO),
% CHECK: nyi-gpu-mailbox-ops.m:[[@LINE-1]]:5: error: 'merk.close_mailbox' op closes a mailbox, which cannot be done in a GPU process yet
    close_mailbox(Mine, !IO).
% CHECK: nyi-gpu-mailbox-ops.m:[[@LINE-1]]:5: error: 'merk.close_mailbox' op closes a mailbox, which cannot be done in a GPU process yet

main(!IO) :-
    new_mailbox(Other, !IO),
    spawn(gpu(0), manager(Other), _, !IO).
