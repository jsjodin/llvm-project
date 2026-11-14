% A receive from a mailbox that another process has closed is an error, as
% process.receive has it: at the host, and in a GPU process, which the host
% tells through the ring it fills for it (device-ring.h, kRingClosed), in
% process-closed-receive-gpu.test. The waiter may be receiving already when
% its mailbox closes, or may start after. At the host the error ends the
% waiter; on the GPU it ends the program.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host 2>&1 | FileCheck %s

:- module process_closed_receive.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

    % Takes words until none comes for half a second, and reports how many.
:- process waiter(mailbox(int)) receives int on gpu.

waiter(Report, Self, !IO) :-
    wait_for(0, Report, Self, !IO).

:- pred wait_for(int::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.

wait_for(N, Report, Self, !IO) :-
    receive(Self, any, after(500), R, !IO),
    ( if R = message(_) then
        wait_for(N + 1, Report, Self, !IO)
    else
        send(Report, N, !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reports, !IO),
    spawn(Place, waiter(Reports), W, !IO),
    close_mailbox(W, !IO),
    send(W, 1, !IO),
    receive(Reports, any, after(2000), R, !IO),
    io.write(R, !IO),
    io.nl(!IO).
% CHECK:Software Error: receive: the mailbox is closed or another node's
