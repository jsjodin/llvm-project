% Throughput of receive goals between the host's processes: how long a
% message takes, from main's first send to the consumer's answer, over
% many messages (README.md).
%
%   receive-goal plain N     one mailbox, every message selected
%   receive-goal skip K N    K messages that no case selects wait in front
%                            of the N, so each receive looks past them
%   receive-goal two N       two mailboxes, the messages alternating
%
% Each prints the messages, their sum, which checks that every one was
% taken once, and the nanoseconds a message.

:- module receive_goal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list, process, string.

:- type job
    --->    job(int)
    ;       noise(int)
    ;       stop.

:- type report
    --->    other(mailbox(job))
    ;       sum(int).

    % Takes jobs from its own mailbox until stop, leaving noise queued.
:- process consumer(mailbox(report)) receives job.

consumer(Report, Self, !IO) :-
    consume(Report, Self, 0, !IO).

:- pred consume(mailbox(report)::in, mailbox(job)::in, int::in,
    io::di, io::uo) is det.

consume(Report, Self, Sum, !IO) :-
    receive [io(!IO)] (
        select(Self, job(I)) ->
            consume(Report, Self, Sum + I, !IO)
    ;
        select(Self, stop) ->
            send(Report, sum(Sum), !IO)
    ).

    % Takes jobs from its own mailbox and one of its own that it gives
    % main, until stop on either.
:- process two_consumer(mailbox(report)) receives job.

two_consumer(Report, Self, !IO) :-
    new_mailbox(Other, !IO),
    send(Report, other(Other), !IO),
    consume_two(Report, Self, Other, 0, !IO).

:- pred consume_two(mailbox(report)::in, mailbox(job)::in, mailbox(job)::in,
    int::in, io::di, io::uo) is det.

consume_two(Report, Self, Other, Sum, !IO) :-
    receive [io(!IO)] (
        select(Self, job(I)) ->
            consume_two(Report, Self, Other, Sum + I, !IO)
    ;
        select(Other, job(J)) ->
            consume_two(Report, Self, Other, Sum + J, !IO)
    ;
        select(Self, stop) ->
            send(Report, sum(Sum), !IO)
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    new_mailbox(Report, !IO),
    ( if
        Args = ["plain", NS],
        string.to_int(NS, N)
    then
        spawn(consumer(Report), Self, !IO),
        now(T0, !IO),
        send_jobs(Self, Self, 1, N, !IO),
        send(Self, stop, !IO),
        finish("plain", N, Report, T0, !IO)
    else if
        Args = ["skip", KS, NS],
        string.to_int(KS, K),
        string.to_int(NS, N)
    then
        spawn(consumer(Report), Self, !IO),
        send_noise(Self, 1, K, !IO),
        now(T0, !IO),
        send_jobs(Self, Self, 1, N, !IO),
        send(Self, stop, !IO),
        finish("skip " ++ KS, N, Report, T0, !IO)
    else if
        Args = ["two", NS],
        string.to_int(NS, N)
    then
        spawn(two_consumer(Report), Self, !IO),
        receive(Report, any, infinity, Got, !IO),
        ( if Got = message(other(Other)) then
            now(T0, !IO),
            send_jobs(Self, Other, 1, N, !IO),
            send(Self, stop, !IO),
            finish("two", N, Report, T0, !IO)
        else
            io.write_string("no mailbox from the consumer\n", !IO)
        )
    else
        io.write_string("usage: receive-goal plain N | skip K N | two N\n",
            !IO)
    ).

    % Sends job(I) to job(N), the odd ones to A and the even ones to B.
:- pred send_jobs(mailbox(job)::in, mailbox(job)::in, int::in, int::in,
    io::di, io::uo) is det.

send_jobs(A, B, I, N, !IO) :-
    ( if I =< N then
        ( if I mod 2 = 1 then send(A, job(I), !IO) else send(B, job(I), !IO) ),
        send_jobs(A, B, I + 1, N, !IO)
    else
        true
    ).

:- pred send_noise(mailbox(job)::in, int::in, int::in, io::di, io::uo)
    is det.

send_noise(Box, I, K, !IO) :-
    ( if I =< K then
        send(Box, noise(I), !IO),
        send_noise(Box, I + 1, K, !IO)
    else
        true
    ).

:- pred finish(string::in, int::in, mailbox(report)::in, int::in,
    io::di, io::uo) is det.

finish(Name, N, Report, T0, !IO) :-
    receive(Report, any, infinity, Got, !IO),
    now(T1, !IO),
    ( if Got = message(sum(Sum)) then
        io.format("%-8s %9d messages, sum %d, %.1f ns a message\n",
            [s(Name), i(N), i(Sum),
             f(float(T1 - T0) / float(max(N, 1)))], !IO)
    else
        io.write_string("no sum from the consumer\n", !IO)
    ).

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(Nanoseconds::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    Nanoseconds = (MR_Integer) ts.tv_sec * 1000000000 + ts.tv_nsec;
").
