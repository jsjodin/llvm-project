% The kernel of a GPU's processes (docs/INPROGRESS.md, GPU processes that run
% loops, one kernel for a GPU's processes): a module's kinds run in its
% image's kernel, merk_place, a wave for each process, which
% process-places-gpu.test runs. A GPU runs one module's kernel at a time: a
% spawn of another module's kind while it runs fails, naming the module that
% holds the GPU. The kernel ends with its last process, and the next spawn
% launches it again. A monitor of a GPU process says when it has ended, after
% which its kernel is ending, so a phase may follow another: a spawn of the
% next module's kind then succeeds. A monitor of a GPU process that has ended
% says noproc. Only the GPU receives from a GPU process's mailbox: a host
% process that does is a software error. A spawn fails, rather than waits,
% when the kernel has no free wave. The share of the GPU's units that its
% kernel takes changes between kernels: the phases alternate all of them and
% half, each kernel on a queue of its own units, and a change while the
% kernel runs a process is a software error, as is one for the host or of a
% share that is no percentage.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 -I %S/Inputs %libomp-flags %S/Inputs/place_phase.m %s -o %t2
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 -I %S/Inputs %libomp-flags %S/Inputs/place_phase.m %s -o %t
% RUN: merkc -fc1 -fprocess-declarations -fprocess-kernels -triple amdgpu10.30-amd-amdhsa -emit-llvm -I %S/Inputs %s -o - | FileCheck %s --check-prefix=DISPATCH
%
% The module's kinds by their numbers, in the order of their declarations.
% DISPATCH:define hidden void @merk_process_dispatch(i64 %0, i64 %1) #{{[0-9]+}} {
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
        spawn(gpu(0), adder(Me), _, !IO),
        spawn(gpu(0), doubler(Me), _, !IO)
    else if Args = ["full"] then
        spawn_waiters(0, !IO)
    else if Args = ["units-running"] then
        spawn(gpu(0), adder(Me), _, !IO),
        set_gpu_units(gpu(0), 50, !IO)
    else if Args = ["units-host"] then
        set_gpu_units(host, 50, !IO)
    else if Args = ["units-percent"] then
        set_gpu_units(gpu(0), 0, !IO)
    else if Args = ["steal"] then
        spawn(gpu(0), adder(Me), A, !IO),
        receive(A, any, after(10), _, !IO)
    else if Args = ["phases"] then
        new_mailbox(Downs, !IO),
        phases(3, 10, Me, Downs, !IO)
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
        spawn(gpu(0), adder(Me), A, !IO),
        send(A, N, !IO),
        receive(Me, any, after(10000), R, !IO),
        ( if R = message(M) then
            io.format("%d\n", [i(M)], !IO),
            adders(Count - 1, M, Me, !IO)
        else
            io.write_string("no answer\n", !IO)
        )
    ).

    % Runs an adder, then a doubler once the adder has ended, Count times.
:- pred phases(int::in, int::in, mailbox(int)::in, mailbox(down_info)::in,
    io::di, io::uo) is det.

phases(Count, N, Me, Downs, !IO) :-
    ( if Count = 0 then
        true
    else
        set_gpu_units(gpu(0), 100, !IO),
        spawn(gpu(0), adder(Me), A, !IO),
        monitor(A, Downs, id, _, !IO),
        send(A, N, !IO),
        answer(Me, Added, !IO),
        ended(Downs, AddedEnd, !IO),
        % The adder's kernel is ending, which a change allows.
        set_gpu_units(gpu(0), 50, !IO),
        spawn(gpu(0), doubler(Me), D, !IO),
        send(D, Added, !IO),
        answer(Me, Doubled, !IO),
        ended(Downs, _, !IO),
        % A's mailbox has closed: its process has ended.
        monitor(A, Downs, id, _, !IO),
        ended(Downs, Again, !IO),
        io.format("%d %s %d %s\n", [i(Added), s(AddedEnd), i(Doubled),
            s(Again)], !IO),
        phases(Count - 1, Doubled, Me, Downs, !IO)
    ).

:- func id(down_info) = down_info.

id(D) = D.

:- pred answer(mailbox(int)::in, int::out, io::di, io::uo) is det.

answer(Me, N, !IO) :-
    receive(Me, any, after(10000), R, !IO),
    ( if R = message(M) then
        N = M
    else
        N = -1
    ).

    % Why the process that a monitor watched ended.
:- pred ended(mailbox(down_info)::in, string::out, io::di, io::uo) is det.

ended(Downs, Reason, !IO) :-
    receive(Downs, any, after(10000), R, !IO),
    ( if R = message(D) then
        Reason = reason_text(reason(D))
    else
        Reason = "none"
    ).

    % Spawns waiters until a spawn fails, which ends the program.
:- pred spawn_waiters(int::in, io::di, io::uo) is det.

spawn_waiters(Count, !IO) :-
    ( if Count > 100000 then
        io.write_string("no spawn failed\n", !IO)
    else
        spawn(gpu(0), waiter, _, !IO),
        spawn_waiters(Count + 1, !IO)
    ).
% CHECK:11
% CHECK-NEXT:12
% CHECK-NEXT:13
% PHASES:11 normal 22 noproc
% PHASES-NEXT:23 normal 46 noproc
% PHASES-NEXT:47 normal 94 noproc
% OTHER:Software Error: process.spawn: GPU 0 runs processes of the module process_places, and a GPU runs one module's processes at a time: those of place_phase may start there once they have ended
% STEAL:Mercury runtime: a process received from a mailbox of another process
% RUNNING:Software Error: process.set_gpu_units: GPU 0 runs processes of the module process_places: the units of its processes' kernels change between kernels, once its processes have all ended
% HOST:Software Error: process.set_gpu_units: the host is not a GPU
% PERCENT:Software Error: process.set_gpu_units: 0 is not a percentage from 1 to 100
% UNSET:Software Error: process.spawn: MERK_GPU_PROCESS_UNITS is not a percentage from 1 to 100: 0
% FULL:Software Error: process.spawn: GPU 0 has no room for another process: the module process_places's kernel there runs {{[1-9][0-9]*}} processes at once, and they are all running
