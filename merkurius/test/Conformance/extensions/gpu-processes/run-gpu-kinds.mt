% DESIGN: GPU processes
% DESIGN: Sized floating point types
% A process kind declared on gpu runs at the place it is spawned at: on
% the host, or on a GPU, where it receives and sends flat messages, ints,
% floats, flat records, strings and arrays of scalars, as the host's
% processes do. It may poll with after(0), and send to another GPU
% process directly. Run with no argument the kinds run at the host, which
% the oracle checks; run-gpu-kinds-gpu.test runs them on the GPU and
% checks the same output. It is a template, run for float and float32.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/gpu_kinds_translation.erl
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fprocess-declarations %t.float.d/run_gpu_kinds_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float.d/run_gpu_kinds_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:message(count(303, 3))
% OUTPUT-FLOAT-NEXT:message(polled("timeout"))
% OUTPUT-FLOAT-NEXT:message(pinged(623))
% OUTPUT-FLOAT-NEXT:message(summed(5050))
% OUTPUT-FLOAT-NEXT:message(summed_floats(3.75))
% OUTPUT-FLOAT-NEXT:message(moved(point(7, 3.0)))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fprocess-declarations %t.float32.d/run_gpu_kinds_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %t.float32.d/run_gpu_kinds_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:message(count(303, 3))
% OUTPUT-FLOAT32-NEXT:message(polled("timeout"))
% OUTPUT-FLOAT32-NEXT:message(pinged(623))
% OUTPUT-FLOAT32-NEXT:message(summed(5050))
% OUTPUT-FLOAT32-NEXT:message(summed_floats(3.75f32))
% OUTPUT-FLOAT32-NEXT:message(moved(point(7, 3.0f32)))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_gpu_kinds.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, @TM@, int, list, process, string.

:- type counter_msg
    --->    add(int)
    ;       urgent(int)
    ;       get(mailbox(report))
    ;       stop.

:- type point
    --->    point(int, @T@).

:- type report
    --->    count(int, int)                 % the count, and the others
    ;       polled(string)
    ;       pinged(int)
    ;       summed(int)
    ;       summed_floats(@T@)
    ;       moved(point).

    % Adds what it is sent, an urgent amount a hundred times, and answers
    % get with the count.
:- process counter(int) receives counter_msg on gpu.

counter(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(add(K)),
        counter(N + K, Self, !IO)
    ;
        R = message(urgent(K)),
        counter(N + K * 100, Self, !IO)
    ;
        R = message(get(Reply)),
        send(Reply, count(N, N mod 100), !IO),
        counter(N, Self, !IO)
    ;
        R = message(stop)
    ;
        R = timeout
    ).

    % Polls its empty mailbox, and reports what it got.
:- process poller(mailbox(report)) receives int on gpu.

poller(Report, Self, !IO) :-
    receive(Self, any, after(0), R, !IO),
    (
        R = message(_),
        send(Report, polled("a message"), !IO)
    ;
        R = timeout,
        send(Report, polled("timeout"), !IO)
    ).

:- type ping
    --->    ping(int, mailbox(int))
    ;       done.

:- process ponger receives ping on gpu.

ponger(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(ping(N, Back)),
        send(Back, N * 2 + 1, !IO),
        ponger(Self, !IO)
    ;
        R = message(done)
    ;
        R = timeout
    ).

    % Pings the ponger Rounds times with what it got back, then reports.
:- process pinger(int, mailbox(ping), mailbox(report)) receives int on gpu.

pinger(Rounds, Ponger, Report, Self, !IO) :-
    pinger_loop(Rounds, 0, Ponger, Report, Self, !IO).

:- pred pinger_loop(int::in, int::in, mailbox(ping)::in, mailbox(report)::in,
    mailbox(int)::in, io::di, io::uo) is det.

pinger_loop(Rounds, N, Ponger, Report, Self, !IO) :-
    ( if Rounds = 0 then
        send(Report, pinged(N), !IO),
        send(Ponger, done, !IO)
    else
        send(Ponger, ping(N, Self), !IO),
        receive(Self, any, infinity, R, !IO),
        (
            R = message(M),
            pinger_loop(Rounds - 1, M mod 1000, Ponger, Report, Self, !IO)
        ;
            R = timeout
        )
    ).

:- type data
    --->    ints(array(int))
    ;       floats(array(@T@))
    ;       move(point, int).

    % Answers each message with what it computes, until it has answered
    % three.
:- process worker(mailbox(report)) receives data on gpu.

worker(Report, Self, !IO) :-
    worker_loop(3, Report, Self, !IO).

:- pred worker_loop(int::in, mailbox(report)::in, mailbox(data)::in,
    io::di, io::uo) is det.

worker_loop(K, Report, Self, !IO) :-
    ( if K = 0 then
        true
    else
        receive(Self, any, infinity, R, !IO),
        (
            R = message(ints(A)),
            send(Report, summed(sum_ints(A, 0, 0)), !IO)
        ;
            R = message(floats(A)),
            send(Report, summed_floats(sum_floats(A, 0, 0.0@S@)), !IO)
        ;
            R = message(move(point(X, Y), D)),
            send(Report, moved(point(X + D, Y * 2.0@S@)), !IO)
        ;
            R = timeout
        ),
        worker_loop(K - 1, Report, Self, !IO)
    ).

:- func sum_ints(array(int), int, int) = int.

sum_ints(A, I, S) =
    ( if I < array.size(A) then sum_ints(A, I + 1, S + array.lookup(A, I))
    else S ).

:- func sum_floats(array(@T@), int, @T@) = @T@.

sum_floats(A, I, S) =
    ( if I < array.size(A) then
        sum_floats(A, I + 1, S + array.lookup(A, I))
    else
        S
    ).

:- pred report(mailbox(report)::in, io::di, io::uo) is det.

report(Reports, !IO) :-
    receive(Reports, any, infinity, R, !IO),
    io.write_line(R, !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    Place = ( if Args = ["gpu"] then gpu(0) else host ),
    new_mailbox(Reports, !IO),
    spawn(Place, counter(0), C, !IO),
    send(C, add(1), !IO),
    send(C, add(2), !IO),
    send(C, urgent(3), !IO),
    send(C, get(Reports), !IO),
    report(Reports, !IO),
    send(C, stop, !IO),
    spawn(Place, poller(Reports), _, !IO),
    report(Reports, !IO),
    spawn(Place, ponger, P, !IO),
    spawn(Place, pinger(50, P, Reports), _, !IO),
    report(Reports, !IO),
    spawn(Place, worker(Reports), W, !IO),
    send(W, ints(array(1 .. 100)), !IO),
    report(Reports, !IO),
    send(W, floats(array([0.5@S@, 1.25@S@, 2.0@S@])), !IO),
    report(Reports, !IO),
    send(W, move(point(4, 1.5@S@), 3), !IO),
    report(Reports, !IO).
