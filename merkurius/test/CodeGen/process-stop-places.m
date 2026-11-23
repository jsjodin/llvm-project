% process.stop and the fault injector at a place (docs/INPROGRESS.md,
% Failure, step 4). On a GPU, a stop sets the process's slot's stop word,
% which the process reads in its receives: it ends there, as at the end of
% its body, back in its kernel's slot loop, which takes another process in
% the slot; its monitors see the reason. The
% first argument is the place, `host` or `gpu`; the second the case.
%
% - "waiter": a process blocked in a receive is stopped: killed.
% - "poller": one that polls its mailbox with no wait, in a loop, is
%   stopped at a poll: killed.
% - "looper": one that runs loops across its worker teams is stopped
%   between them: killed; then another takes the teams it had.
% - "again": after a stop, ten processes more start and end, so a stopped
%   process's slot does not stop the kernel.
% - "injected": MERK_FAULTS fails a process of the kind: failed("injected").
% - "kernel": MERK_FAULTS ends every process of a GPU's kernel at once:
%   three processes, each kernel_ended.
%
% The kinds run at the host here, and on the GPU in
% process-stop-places-gpu.test. mmc has no processes, so the outputs are
% written by hand.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fopenmp -fdo-loops -fprocess-declarations --offload-arch=%offload-archs %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host waiter | FileCheck %s --check-prefix=WAITER --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host poller | FileCheck %s --check-prefix=POLLER --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host looper | FileCheck %s --check-prefix=LOOPER --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host again | FileCheck %s --check-prefix=AGAIN --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_FAULTS="seed=1 mtbf=50 fail=order count=1" %t2 host injected 2>/dev/null | FileCheck %s --check-prefix=INJECTED --match-full-lines --strict-whitespace

% WAITER:waiter: killed
% WAITER-NOT:{{.}}
% POLLER:polled
% POLLER-NEXT:poller: killed
% POLLER-NOT:{{.}}
% LOOPER:looped 500500
% LOOPER-NEXT:looper: killed
% LOOPER-NEXT:looped 500500
% LOOPER-NEXT:second looper: normal
% LOOPER-NOT:{{.}}
% AGAIN:waiter: killed
% AGAIN-NEXT:10 more: normal
% AGAIN-NOT:{{.}}
% INJECTED:poller: injected
% INJECTED-NOT:{{.}}
% KERNEL:kernel_ended
% KERNEL-NEXT:kernel_ended
% KERNEL-NEXT:kernel_ended
% KERNEL-NOT:{{.}}
% KERNELLOG:# seed=1
% KERNELLOG-NEXT:{{[0-9]+}} kernel 0
% FREED-NEXT:Merkurius runtime: 0 messages are neither read nor freed
% FREED-NEXT:Merkurius runtime: 0 chunks of the devices' arenas and 0 rings are allocated
% FREED-NEXT:Merkurius runtime: 0 blocks of messages between GPU processes are allocated

:- module process_stop_places.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type order
    --->    go
    ;       quit.

:- type progress
    --->    polled
    ;       looped(int).

    % What the main process reads: progress from the GPU processes, which
    % must be flat, and their downs, from another mailbox.
:- type report
    --->    progress(progress)
    ;       went(string, down_info).

    % The main process's mailboxes.
:- type boxes
    --->    boxes(mailbox(progress), mailbox(report)).

:- func went_as(string, down_info) = report.
went_as(Name, Down) = went(Name, Down).

:- process waiter receives order on gpu.

waiter(Self, !IO) :-
    receive(Self, any, infinity, _ : received(order), !IO).

    % Polls with no wait until a quit, which never comes; tells the main
    % process once, at its first poll.
:- process poller(mailbox(progress), int) receives order on gpu.

poller(Main, N, Self, !IO) :-
    receive(Self, any, after(0), Got, !IO),
    ( if N = 0 then
        send(Main, polled, !IO)
    else
        true
    ),
    ( if Got = message(quit) then
        true
    else
        poller(Main, N + 1, Self, !IO)
    ).

    % Runs a loop across its worker teams, tells the main process its sum,
    % and waits for an order: quit ends it, go runs the loop again.
:- process looper(mailbox(progress)) receives order on gpu.

looper(Main, Self, !IO) :-
    send(Main, looped(sum(1000)), !IO),
    receive(Self, any, infinity, Got, !IO),
    ( if Got = message(go) then
        looper(Main, Self, !IO)
    else
        true
    ).

:- func sum(int) = int.
sum(N) = S :-
    %$omp teams distribute parallel do reduction((+) : S)
    ( for(I, 1, N), fromto(0, S0, S1, S) do S1 = S0 + I ).

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if
        Arguments = [Name, Case],
        ( Name = "host", Where = host
        ; Name = "gpu", Where = gpu(0)
        )
    then
        new_mailbox(Progress, !IO),
        new_mailbox(Downs, !IO),
        run(Where, Case, boxes(Progress, Downs), !IO),
        close_mailbox(Progress, !IO),
        close_mailbox(Downs, !IO)
    else
        io.write_string("usage: host|gpu CASE\n", !IO)
    ).

:- pred run(place::in, string::in, boxes::in, io::di, io::uo) is det.

run(Where, Case, Main, !IO) :-
    Main = boxes(Progress, Downs),
    ( if Case = "waiter" then
        stop_waiter(Where, Main, !IO)
    else if Case = "poller" then
        spawn(Where, poller(Progress, 0), Poller, !IO),
        monitor(Poller, Downs, went_as("poller"), _, !IO),
        show(Main, !IO),
        stop(Poller, !IO),
        show(Main, !IO)
    else if Case = "looper" then
        set_worker_teams(Where, share(2), !IO),
        spawn(Where, looper(Progress), Looper, !IO),
        monitor(Looper, Downs, went_as("looper"), _, !IO),
        show(Main, !IO),
        stop(Looper, !IO),
        show(Main, !IO),
        spawn(Where, looper(Progress), Second, !IO),
        monitor(Second, Downs, went_as("second looper"), _, !IO),
        show(Main, !IO),
        send(Second, quit, !IO),
        show(Main, !IO)
    else if Case = "again" then
        stop_waiter(Where, Main, !IO),
        more(Where, Downs, 10, !IO),
        io.write_string("10 more: normal\n", !IO)
    else if Case = "injected" then
        spawn(Where, poller(Progress, 0), Poller, !IO),
        monitor(Poller, Downs, went_as("poller"), _, !IO),
        wait_down(Downs, !IO)
    else if Case = "kernel" then
        list.foldl(watched_waiter(Where, Downs), 1 .. 3, !IO),
        wait_down(Downs, !IO),
        wait_down(Downs, !IO),
        wait_down(Downs, !IO)
    else
        io.write_string("no case " ++ Case ++ "\n", !IO)
    ).

:- pred stop_waiter(place::in, boxes::in, io::di, io::uo) is det.

stop_waiter(Where, Main, !IO) :-
    Main = boxes(_, Downs),
    spawn(Where, waiter, Waiter, !IO),
    monitor(Waiter, Downs, went_as("waiter"), _, !IO),
    stop(Waiter, !IO),
    show(Main, !IO).

:- pred watched_waiter(place::in, mailbox(report)::in, int::in,
    io::di, io::uo) is det.

watched_waiter(Where, Main, _, !IO) :-
    spawn(Where, waiter, Waiter, !IO),
    monitor(Waiter, Main, (func(D) = went("", D)), _, !IO).

    % N waiters, each told to go, each of which must end normally.
:- pred more(place::in, mailbox(report)::in, int::in, io::di, io::uo) is det.

more(Where, Main, N, !IO) :-
    ( if N = 0 then
        true
    else
        spawn(Where, waiter, Waiter, !IO),
        monitor(Waiter, Main, went_as("more"), _, !IO),
        send(Waiter, go, !IO),
        receive(Main, any, after(30000), Got, !IO),
        ( if Got = message(went(_, Down)), reason(Down) = normal then
            more(Where, Main, N - 1, !IO)
        else
            io.write_string("a waiter did not end normally\n", !IO)
        )
    ).

    % Shows the next report: a down with its name, or, for an unnamed one,
    % its reason alone.
:- pred show(boxes::in, io::di, io::uo) is det.

show(boxes(Progress, Downs), !IO) :-
    Selector = selecting(Progress, (func(P) = progress(P)),
        selecting(Downs, (func(R) = R), new_selector)),
    receive(Selector, after(30000), Got, !IO),
    (
        Got = message(progress(polled)),
        io.write_string("polled\n", !IO)
    ;
        Got = message(progress(looped(S))),
        io.format("looped %d\n", [i(S)], !IO)
    ;
        Got = message(went(Name, Down)),
        ( if Name = "" then
            io.write_string(reason_text(reason(Down)) ++ "\n", !IO)
        else
            io.write_string(Name ++ ": " ++ reason_text(reason(Down)) ++ "\n",
                !IO)
        )
    ;
        Got = timeout,
        io.write_string("nothing\n", !IO)
    ).

    % The next down, skipping the poller's report.
:- pred wait_down(mailbox(report)::in, io::di, io::uo) is det.

wait_down(Main, !IO) :-
    receive(Main, (pred(went(N, D)::in, went(N, D)::out) is semidet),
        after(30000), Got, !IO),
    ( if Got = message(Report) then
        receive_shown(Report, !IO)
    else
        io.write_string("nothing\n", !IO)
    ).

:- pred receive_shown(report::in, io::di, io::uo) is det.

receive_shown(Report, !IO) :-
    ( if Report = went(Name, Down) then
        ( if Name = "" then
            io.write_string(reason_text(reason(Down)) ++ "\n", !IO)
        else
            io.write_string(Name ++ ": " ++ reason_text(reason(Down)) ++ "\n",
                !IO)
        )
    else
        true
    ).
