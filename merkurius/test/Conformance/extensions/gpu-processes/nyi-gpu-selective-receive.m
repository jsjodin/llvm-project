% DESIGN: GPU processes / The device runtime
% A receive with a selector other than any, which takes the oldest message
% the selector accepts, needs a loop over the mailbox's messages that calls
% the selector, which a GPU process cannot make yet: reported at the
% receive.
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
% CHECK: nyi-gpu-selective-receive.m:[[@LINE-1]]:5: error: 'merk.receive' op receives with a selector other than any, which cannot be done in a GPU process yet
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
