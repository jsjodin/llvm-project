% The kernel of a GPU's processes (docs/INPROGRESS.md, GPU processes that run
% loops, one kernel for a GPU's processes): a module's kinds run in its
% image's kernel, merk_place, a wave for each process, which
% process-places-gpu.test runs. A GPU runs one module's kernel at a time: a
% spawn of another module's kind while it runs fails, naming the module that
% holds the GPU. The kernel ends with its last process, and the next spawn
% launches it again. A spawn fails, rather than waits, when the kernel has no
% free wave.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 -I %S/Inputs %libomp-flags %S/Inputs/place_phase.m %s -o %t2
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 -I %S/Inputs %libomp-flags %S/Inputs/place_phase.m %s -o %t
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm -I %S/Inputs %s -o - | FileCheck %s --check-prefix=DISPATCH
%
% The module's kinds by their numbers, in the order of their declarations.
% DISPATCH:define hidden void @merk_process_dispatch(i64 %0, i64 %1) {
% DISPATCH-NEXT:  switch i64 %0, label %5 [
% DISPATCH-NEXT:    i64 0, label %3
% DISPATCH-NEXT:    i64 1, label %4
% DISPATCH-NEXT:  ]
% DISPATCH:  call void @merk_process_adder(i64 %1)
% DISPATCH:  call void @merk_process_waiter(i64 %1)
:- module process_places.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, place_phase, process, string.

    % Sends Reply the number it receives, plus one, and ends.
:- process adder(mailbox(int)) receives int on gpu.

adder(Reply, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(N) then
        send(Reply, N + 1, !IO)
    else
        true
    ).

    % Waits for one message, which never comes.
:- process waiter receives int on gpu.

waiter(Self, !IO) :-
    receive(Self, any, infinity, _, !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Me, !IO),
    ( if Args = ["other"] then
        % adder holds the GPU while it waits.
        spawn_adder_at(gpu(0), Me, _, !IO),
        spawn_doubler_at(gpu(0), Me, _, !IO)
    else if Args = ["full"] then
        spawn_waiters(0, !IO)
    else
        % Each adder answers and ends: the next runs in the same kernel, or
        % in a new one if that kernel's last process had ended.
        adders(3, 10, Me, !IO)
    ).

:- pred adders(int::in, int::in, mailbox(int)::in, io::di, io::uo) is det.

adders(Count, N, Me, !IO) :-
    ( if Count = 0 then
        true
    else
        spawn_adder_at(gpu(0), Me, A, !IO),
        send(A, N, !IO),
        receive(Me, any, after(10000), R, !IO),
        ( if R = message(M) then
            io.format("%d\n", [i(M)], !IO),
            adders(Count - 1, M, Me, !IO)
        else
            io.write_string("no answer\n", !IO)
        )
    ).

    % Spawns waiters until a spawn fails, which ends the program.
:- pred spawn_waiters(int::in, io::di, io::uo) is det.

spawn_waiters(Count, !IO) :-
    ( if Count > 100000 then
        io.write_string("no spawn failed\n", !IO)
    else
        spawn_waiter_at(gpu(0), _, !IO),
        spawn_waiters(Count + 1, !IO)
    ).
% CHECK:11
% CHECK-NEXT:12
% CHECK-NEXT:13
% OTHER:Software Error: place_phase.spawn_doubler_at: GPU 0 runs processes of the module process_places, and a GPU runs one module's processes at a time: those of place_phase may start there once they have ended
% FULL:Software Error: process_places.spawn_waiter_at: GPU 0 has no room for another process: the module process_places's kernel there runs {{[1-9][0-9]*}} processes at once, and they are all running
