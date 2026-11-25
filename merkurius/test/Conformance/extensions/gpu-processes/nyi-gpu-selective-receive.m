% DESIGN: GPU processes / The device runtime
% A receive with a selector other than any, which takes the oldest message
% the selector accepts, needs a closure, which a GPU process cannot make
% or call yet: reported at the selector, with a note for each call back to
% the module's own code.
% ORACLE: none
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: not merkc -O2 -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t 2>&1 | FileCheck %s --implicit-check-not=error:
:- module nyi_gpu_selective_receive.
:- interface.
:- import_module io, process.
:- pred main(io::di, io::uo) is det.

:- type msg
    --->    data(int)
    ;       urgent(int).

:- process taker(mailbox(int)) receives msg on gpu.

:- implementation.

taker(Report, Self, !IO) :-
    receive(Self, is_urgent, infinity, R, !IO),
% CHECK: nyi-gpu-selective-receive.m:[[@LINE-1]]:19: error: 'merk.closure' op makes a closure, which cannot be done in a GPU process yet
% CHECK: process.m:{{[0-9]+}}:{{[0-9]+}}: error: 'merk.call_closure' op calls a closure, which cannot be done in a GPU process yet
% CHECK: nyi-gpu-selective-receive.m:[[@LINE-3]]:5: note: called by nyi_gpu_selective_receive.taker_4_0 here: process.receive_6_0
% CHECK: process.m:{{[0-9]+}}:{{[0-9]+}}: error: 'merk.call_closure' op calls a closure, which cannot be done in a GPU process yet
% CHECK: nyi-gpu-selective-receive.m:[[@LINE-5]]:5: note: called by nyi_gpu_selective_receive.taker_4_0 here: process.receive_6_0
    (
        R = message(N),
        send(Report, N, !IO)
    ;
        R = timeout
    ).

:- pred is_urgent(msg::in, int::out) is semidet.

is_urgent(urgent(N), N).

main(!IO) :-
    new_mailbox(Report, !IO),
    spawn(gpu(0), taker(Report), T, !IO),
    send(T, data(1), !IO),
    send(T, urgent(2), !IO),
    receive(Report, any, after(1000), R, !IO),
    io.write_line(R, !IO).
