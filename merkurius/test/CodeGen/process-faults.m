% The fault injector (docs/INPROGRESS.md, Failure, step 3): MERK_FAULTS fails
% processes of a kind, named by their mailbox's message type, at random
% times from a seed, and writes the schedule, which a later run replays. A
% supervisor restarts each worker that fails; some workers block, others
% spin, and a process of another kind is never hit. mmc has no processes,
% so the output is written by hand.
%
% RUN: merkc -O2 %s -o %t
% RUN: env MERK_FAULTS="seed=7 mtbf=20 fail=job count=6 log=%t.schedule" %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: FileCheck %s --check-prefix=SCHEDULE --match-full-lines --strict-whitespace < %t.schedule
% SCHEDULE:# seed=7
% SCHEDULE-COUNT-6:{{[0-9]+}} fail job {{[0-9]+}}
% SCHEDULE-NOT:{{.}}
%
% The replay injects the same faults at the same times.
% RUN: env MERK_FAULTS="replay=%t.schedule log=%t.replayed" %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: tail -n +2 %t.schedule | diff - %t.replayed
%
% A qualified kind, one worker, -O0, and every message read or freed.
% RUN: merkc -O0 %s -o %t.0
% RUN: env MERK_WORKERS=1 MERK_FAULTS="seed=3 mtbf=10 fail=process_faults.job count=6" %t.0 2>&1 | FileCheck %s --check-prefixes=OUTPUT,STDERR --match-full-lines --strict-whitespace
% STDERR:Merkurius runtime: fault: # seed=3
% STDERR-COUNT-6:Merkurius runtime: fault: {{[0-9]+}} fail process_faults.job {{[0-9]+}}
% RUN: env MERK_CHECK_MESSAGES=1 MERK_FAULTS="seed=5 mtbf=10 fail=job count=6 log=%t.check" %t 2>&1 >/dev/null | FileCheck %s --check-prefix=MESSAGES --match-full-lines
% MESSAGES:Merkurius runtime: 0 messages are neither read nor freed
%
% Settings that are wrong.
% RUN: not env MERK_FAULTS="mtbf=10" %t 2>&1 | FileCheck %s --check-prefix=NOTHING
% NOTHING:Merkurius runtime: MERK_FAULTS: no faults to inject: fail=KINDS or partition
% RUN: not env MERK_FAULTS="fail=job" %t 2>&1 | FileCheck %s --check-prefix=NOMTBF
% NOMTBF:Merkurius runtime: MERK_FAULTS: mtbf=MILLISECONDS is missing
% RUN: not env MERK_FAULTS="mtbf=10 fail=job often" %t 2>&1 | FileCheck %s --check-prefix=UNKNOWN
% UNKNOWN:Merkurius runtime: MERK_FAULTS: not a setting: often
% RUN: not env MERK_FAULTS="mtbf=10 kernel" %t 2>&1 | FileCheck %s --check-prefix=KERNEL
% KERNEL:Merkurius runtime: MERK_FAULTS: ending a GPU kernel's processes is not supported yet
% RUN: not env MERK_FAULTS="replay=%t.schedule mtbf=10" %t 2>&1 | FileCheck %s --check-prefix=REPLAY
% REPLAY:Merkurius runtime: MERK_FAULTS: replay takes no other settings but log

% OUTPUT:failed: injected
% OUTPUT-NEXT:failed: injected
% OUTPUT-NEXT:failed: injected
% OUTPUT-NEXT:failed: injected
% OUTPUT-NEXT:failed: injected
% OUTPUT-NEXT:failed: injected
% OUTPUT-NEXT:the bystander: killed
% OUTPUT-NOT:{{.}}

:- module process_faults.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type job
    --->    job(int).

:- type happening
    --->    worker_down(down_info)
    ;       bystander_down(down_info).

:- func worker_down_as(down_info) = happening.
worker_down_as(Down) = worker_down(Down).

:- func bystander_down_as(down_info) = happening.
bystander_down_as(Down) = bystander_down(Down).

    % Even workers wait for a job that never comes; odd ones spin.
    %
:- pred worker(int::in, mailbox(job)::in, io::di, io::uo) is det.

worker(N, Box, !IO) :-
    ( if N mod 2 = 0 then
        receive(Box, any, infinity, _ : received(job), !IO)
    else
        spin(1000000000000000000, N, M),
        io.write_int(M, !IO)
    ).

    % A random-number recurrence that the optimizer cannot shortcut.
    %
:- pred spin(int::in, int::in, int::out) is det.

spin(K, X, M) :-
    ( if K = 0 then
        M = X
    else
        spin(K - 1, X * 6364136223846793005 + 1442695040888963407, M)
    ).

:- pred bystander(mailbox(string)::in, io::di, io::uo) is det.

bystander(Box, !IO) :-
    receive(Box, any, infinity, _ : received(string), !IO).

:- pred start_worker(mailbox(happening)::in, int::in, io::di, io::uo) is det.

start_worker(Events, N, !IO) :-
    spawn(worker(N), Worker, !IO),
    monitor(Worker, Events, worker_down_as, _, !IO).

    % Restarts each worker that goes down, until Left have.
    %
:- pred supervise(mailbox(happening)::in, int::in, int::in, io::di, io::uo)
    is det.

supervise(Events, Next, Left, !IO) :-
    ( if Left = 0 then
        true
    else
        receive(Events, any, after(60000), Got, !IO),
        ( if Got = message(worker_down(Down)) then
            io.write_string("failed: " ++ reason_text(reason(Down)) ++ "\n",
                !IO),
            start_worker(Events, Next, !IO),
            supervise(Events, Next + 1, Left - 1, !IO)
        else
            io.write_string("no down\n", !IO)
        )
    ).

main(!IO) :-
    new_mailbox(Events, !IO),
    spawn(bystander, Bystander, !IO),
    monitor(Bystander, Events, bystander_down_as, _, !IO),
    list.foldl(start_worker(Events), 1 .. 4, !IO),
    supervise(Events, 5, 6, !IO),
    stop(Bystander, !IO),
    receive(Events, (pred(bystander_down(D)::in, D::out) is semidet),
        after(10000), Got, !IO),
    ( if Got = message(Down) then
        io.write_string("the bystander: " ++ reason_text(reason(Down)) ++
            "\n", !IO)
    else
        io.write_string("the bystander: nothing\n", !IO)
    ),
    close_mailbox(Events, !IO).
