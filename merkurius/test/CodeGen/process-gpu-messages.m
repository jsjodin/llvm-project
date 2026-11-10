% Messages of any type between the host and process kinds that run on a GPU
% (docs/INPROGRESS.md, GPU processes without omp target, step 4e). A
% message that is not one word, nor an array of one-word elements, is
% copied by its type_info (runtime/copy.h): from the GPU into host memory,
% which the host sends on, and from the host's mailbox into host memory that
% the GPU copies into its arena. A counter on the GPU takes add(N) and
% get(Reply); two processes on the GPU play ping-pong through the host's
% mailboxes; and a summer takes a list and a string and answers with the
% string and the list's sum. A closure cannot cross, as the GPU has other
% procedures. With "many", two hundred summers each take one message of the
% two they are sent and end, after which none of the host's copies of their
% messages is left (MERK_CHECK_PINNED), nor any of the messages, those the
% GPU took and those it did not (MERK_CHECK_MESSAGES). The program runs its kinds at the host here, and at the GPU in
% process-gpu-messages-gpu.test.
%
% REQUIRES: amdgpu-registered-target, libomp
% RUN: merkc -O2 -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t2
% RUN: %libomp-env %t2 host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -fprocess-declarations --offload-arch=gfx1030 %libomp-flags %s -o %t
% RUN: %libomp-env %t host | FileCheck %s --match-full-lines --strict-whitespace
% RUN: %libomp-env %t2 host closure | FileCheck %s --check-prefix=CLOSURE --match-full-lines --strict-whitespace
% RUN: %libomp-env MERK_CHECK_MESSAGES=1 %t2 host many 2>&1 | FileCheck %s --check-prefixes=MANY,MESSAGES --match-full-lines --strict-whitespace

:- module process_gpu_messages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, process, string.

:- type counter_msg
    --->    add(int)
    ;       get(mailbox(int))
    ;       stop.

    % A counter that starts at N.
:- process counter(int) receives counter_msg on gpu.

counter(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(add(M)) then
        counter(N + M, Self, !IO)
    else if R = message(get(Reply)) then
        send(Reply, N, !IO),
        counter(N, Self, !IO)
    else
        true
    ).

:- type ping
    --->    ping(int, mailbox(int))
    ;       done.

    % Answers each ping(N, Back) with N + 1 to Back, until done.
:- process ponger receives ping on gpu.

ponger(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(ping(N, Back)) then
        send(Back, N + 1, !IO),
        ponger(Self, !IO)
    else
        true
    ).

    % Pings the ponger with what it last got back, Rounds times, then
    % reports the last answer and tells the ponger it is done.
:- process pinger(int, mailbox(ping), mailbox(report)) receives int on gpu.

:- type report
    --->    report(string, int).

pinger(Rounds, Ponger, Report, Self, !IO) :-
    pinger_loop(Rounds, 0, Ponger, Report, Self, !IO).

:- pred pinger_loop(int::in, int::in, mailbox(ping)::in, mailbox(report)::in,
    mailbox(int)::in, io::di, io::uo) is det.

pinger_loop(Rounds, N, Ponger, Report, Self, !IO) :-
    ( if Rounds = 0 then
        send(Report, report("pinged", N), !IO),
        send(Ponger, done, !IO)
    else
        send(Ponger, ping(N, Self), !IO),
        receive(Self, any, after(10000), R, !IO),
        ( if R = message(M) then
            pinger_loop(Rounds - 1, M, Ponger, Report, Self, !IO)
        else
            send(Report, report("no answer", N), !IO)
        )
    ).

:- type sum_request
    --->    sum(list(int), string, mailbox(report)).

    % Answers sum(Numbers, Name, Reply) with the name and the numbers' sum,
    % once.
:- process summer receives sum_request on gpu.

summer(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    ( if R = message(sum(Numbers, Name, Reply)) then
        send(Reply, report(Name, sum(Numbers)), !IO)
    else
        true
    ).

    % Without a closure, which a GPU process cannot make yet.
:- func sum(list(int)) = int.

sum([]) = 0.
sum([N | Ns]) = N + sum(Ns).

:- type closure_msg
    --->    use(pred(int, int)).

    % Says when it has its message, which at a GPU it never has: the
    % closure cannot cross.
:- process applier(mailbox(int)) receives closure_msg on gpu.

applier(Back, Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    send(Back, 1, !IO).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [Where | Rest] then
        Place = ( if Where = "gpu" then gpu(0) else host ),
        ( if Rest = ["many"] then
            new_mailbox(Reports, !IO),
            many(Place, 200, 0, Total, Reports, !IO),
            io.write_int(Total, !IO), io.nl(!IO)
        else if Rest = ["closure"] then
            new_mailbox(Back, !IO),
            spawn_applier_at(Place, Back, A, !IO),
            send(A, use(twice), !IO),
            receive(Back, any, after(10000), R, !IO),
            io.write(R, !IO), io.nl(!IO)
        else
            run(Place, !IO)
        )
    else
        io.write_string("usage: host or gpu, then closure or nothing\n", !IO)
    ).

:- pred many(place::in, int::in, int::in, int::out, mailbox(report)::in,
    io::di, io::uo) is det.

many(Place, N, !Total, Reports, !IO) :-
    ( if N > 0 then
        spawn_summer_at(Place, S, !IO),
        send(S, sum([N, 1], "many", Reports), !IO),
        send(S, sum([N], "unread", Reports), !IO),
        receive(Reports, any, after(10000), R, !IO),
        ( if R = message(report(_, Sum)) then
            !:Total = !.Total + Sum
        else
            true
        ),
        many(Place, N - 1, !Total, Reports, !IO)
    else
        true
    ).

:- pred twice(int::in, int::out) is det.

twice(N, 2 * N).

:- pred run(place::in, io::di, io::uo) is det.

run(Place, !IO) :-
    new_mailbox(Ints, !IO),
    spawn_counter_at(Place, 10, C, !IO),
    send(C, add(5), !IO),
    send(C, get(Ints), !IO),
    send(C, add(27), !IO),
    send(C, get(Ints), !IO),
    send(C, stop, !IO),
    receive(Ints, any, after(10000), R1, !IO),
    receive(Ints, any, after(10000), R2, !IO),
    io.write(R1, !IO), io.nl(!IO),
    io.write(R2, !IO), io.nl(!IO),
    new_mailbox(Reports, !IO),
    spawn_ponger_at(Place, P, !IO),
    spawn_pinger_at(Place, 100, P, Reports, _, !IO),
    receive(Reports, any, after(10000), R3, !IO),
    io.write(R3, !IO), io.nl(!IO),
    spawn_summer_at(Place, S, !IO),
    send(S, sum(1 .. 100, "one to a hundred", Reports), !IO),
    receive(Reports, any, after(10000), R4, !IO),
    io.write(R4, !IO), io.nl(!IO).
% CHECK:message(15)
% CHECK-NEXT:message(42)
% CHECK-NEXT:message(report("pinged", 100))
% CHECK-NEXT:message(report("one to a hundred", 5050))
% CLOSURE:message(1)
% MANY:20300
% MESSAGES-NEXT:Mercury runtime: 0 messages are neither read nor freed
