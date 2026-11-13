% A program that ends by C's exit, from a foreign_proc, while a process
% still waits (runtime/offload.cpp, EndWhileRegionsRun). A GPU process or a
% target region still runs its kernel then, and the exit handlers of
% liboffload and libomptarget freed the device memory it used: the kernel
% faulted, the plugin aborted the program, and neither the status nor the
% buffered output survived. The program now ends at once, with its status,
% as when main ends first. The waiter is a host process here, and a GPU
% process or one that waits in a target region on the GPU in
% process-exit-running-gpu.test.
%
% mmc has no process module to run it with, so the output is written by
% hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -fopenmp -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env sh -c '%t host; echo "exit status $?"' | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env sh -c '%t2 host; echo "exit status $?"' | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations %s -o %t.sequential
% RUN: sh -c '%t.sequential region; echo "exit status $?"' | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t.sequential.O2
% RUN: sh -c '%t.sequential.O2 region; echo "exit status $?"' | FileCheck %s --match-full-lines --strict-whitespace

% CHECK:the waiter started: 1
% CHECK-NEXT:exit status 3

:- module process_exit_running.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, process, string.

:- type to_waiter
    --->    stop.

    % Tells the host it has started, then waits for a message that never
    % comes.
:- process waiter(mailbox(int)) receives to_waiter on gpu.
waiter(Host, Self, !IO) :-
    send(Host, 1, !IO),
    receive(Self, any, infinity, _, !IO).

    % The same in a target region.
:- process region_waiter(mailbox(int)) receives int.
region_waiter(Host, Self, !IO) :-
    %$omp target
    (
        send(Host, 1, !IO),
        receive(Self, any, infinity, _, !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Me, !IO),
    ( if Args = ["region"] then
        spawn(host, region_waiter(Me), _, !IO)
    else
        Place = ( if Args = ["gpu"] then gpu(0) else host ),
        spawn(Place, waiter(Me), _, !IO)
    ),
    receive(Me, any, infinity, R, !IO),
    ( if R = message(N) then
        io.format("the waiter started: %d\n", [i(N)], !IO)
    else
        true
    ),
    c_exit(3, !IO).

:- pragma foreign_decl("C", "#include <stdlib.h>").

:- pred c_exit(int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    c_exit(Status::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    exit((int) Status);
").
