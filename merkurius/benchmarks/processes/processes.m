% What a process costs (docs/INPROGRESS.md, The scheduler, step 3).
%
%   processes pingpong ROUNDS      two processes pass a message back and forth
%   processes work COUNT SIZE      COUNT processes each compute fib(SIZE)
%   processes pairs COUNT ROUNDS   COUNT pairs of processes play ping-pong
%   processes spawn COUNT [BYTES]  COUNT processes, each of which waits for a
%                                  message, are spawned, all told to go, and
%                                  all have ended; BYTES is their stack's size

:- module processes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, float, list, maybe, process, string.

:- pragma foreign_decl("C", local, "#include <time.h>").

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    T = (MR_Integer) t.tv_sec * 1000000000 + t.tv_nsec;
").

:- pred resident_kib(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C", resident_kib(K::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    long pages = 0, resident = 0;
    FILE *f = fopen(\"/proc/self/statm\", \"r\");
    if (f) {
        fscanf(f, \"%ld %ld\", &pages, &resident);
        fclose(f);
    }
    K = resident * 4;
").
:- pragma foreign_decl("C", local, "#include <stdio.h>").

:- type ball
    --->    ball(int, mailbox(ball)).

:- pred partner(mailbox(ball)::in, io::di, io::uo) is det.
partner(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    ( if Got = yes(ball(N, Back)) then
        send(Back, ball(N, Box), !IO),
        ( if N > 0 then partner(Box, !IO) else true )
    else
        true
    ).

:- pred rounds(int::in, mailbox(ball)::in, mailbox(ball)::in,
    io::di, io::uo) is det.
rounds(N, Other, Mine, !IO) :-
    send(Other, ball(N, Mine), !IO),
    receive(Mine, any, infinity, Got, !IO),
    ( if Got = yes(_), N > 0 then
        rounds(N - 1, Other, Mine, !IO)
    else
        true
    ).

:- pred waiter(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
waiter(Done, Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    ( if Got = yes(N) then send(Done, N, !IO) else true ).

:- pred spawn_all(int::in, int::in, mailbox(int)::in,
    list(mailbox(int))::in, list(mailbox(int))::out, io::di, io::uo) is det.
spawn_all(N, Bytes, Done, !Boxes, !IO) :-
    ( if N = 0 then
        true
    else
        ( if Bytes > 0 then
            spawn_with_stack(Bytes, waiter(Done), Box, !IO)
        else
            spawn(waiter(Done), Box, !IO)
        ),
        !:Boxes = [Box | !.Boxes],
        spawn_all(N - 1, Bytes, Done, !Boxes, !IO)
    ).

:- pred tell_all(list(mailbox(int))::in, io::di, io::uo) is det.
tell_all([], !IO).
tell_all([Box | Boxes], !IO) :-
    send(Box, 1, !IO),
    tell_all(Boxes, !IO).

:- pred collect(int::in, mailbox(int)::in, io::di, io::uo) is det.
collect(N, Done, !IO) :-
    ( if N = 0 then
        true
    else
        receive(Done, any, infinity, _, !IO),
        collect(N - 1, Done, !IO)
    ).

:- pred yield(mailbox(int)::in, io::di, io::uo) is det.
yield(Box, !IO) :-
    receive(Box, any, after(1), _, !IO).

:- func us(int, int) = string.
us(Nanoseconds, Count) =
    string.format("%.3f", [f(float(Nanoseconds) / 1000.0 / float(Count))]).

    % Work and a reply, for each of Count processes.
:- func fib(int) = int.
fib(N) = (if N < 2 then N else fib(N - 1) + fib(N - 2)).

:- pred worker(int::in, mailbox(int)::in, mailbox(int)::in,
    io::di, io::uo) is det.
worker(Size, Done, _, !IO) :-
    send(Done, fib(Size), !IO).

:- pred start_workers(int::in, int::in, mailbox(int)::in,
    io::di, io::uo) is det.
start_workers(N, Size, Done, !IO) :-
    ( if N = 0 then
        true
    else
        spawn(worker(Size, Done), _, !IO),
        start_workers(N - 1, Size, Done, !IO)
    ).

:- pred start_pairs(int::in, int::in, mailbox(int)::in,
    io::di, io::uo) is det.
start_pairs(N, Rounds, Done, !IO) :-
    ( if N = 0 then
        true
    else
        spawn(pair_leader(Rounds, Done), _, !IO),
        start_pairs(N - 1, Rounds, Done, !IO)
    ).

:- pred pair_leader(int::in, mailbox(int)::in, mailbox(ball)::in,
    io::di, io::uo) is det.
pair_leader(Rounds, Done, Mine, !IO) :-
    spawn(partner, Other, !IO),
    rounds(Rounds - 1, Other, Mine, !IO),
    send(Done, 1, !IO).

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if
        Arguments = ["work", CountText, SizeText],
        string.to_int(CountText, Count),
        string.to_int(SizeText, Size)
    then
        new_mailbox(Done, !IO),
        now(T0, !IO),
        start_workers(Count, Size, Done, !IO),
        collect(Count, Done, !IO),
        now(T1, !IO),
        io.write_string(string.format("work %d processes of fib(%d): %s ms\n",
            [i(Count), i(Size), s(us(T1 - T0, 1000))]), !IO)
    else if
        Arguments = ["pairs", CountText, RoundsText],
        string.to_int(CountText, Count),
        string.to_int(RoundsText, Rounds)
    then
        new_mailbox(Done, !IO),
        now(T0, !IO),
        start_pairs(Count, Rounds, Done, !IO),
        collect(Count, Done, !IO),
        now(T1, !IO),
        io.write_string(string.format("pairs %d of %d rounds: %s us a round in all\n",
            [i(Count), i(Rounds), s(us(T1 - T0, Count * Rounds))]), !IO)
    else if
        Arguments = ["pingpong", Text],
        string.to_int(Text, Rounds)
    then
        spawn(partner, Other, !IO),
        new_mailbox(Mine, !IO),
        now(T0, !IO),
        rounds(Rounds - 1, Other, Mine, !IO),
        now(T1, !IO),
        io.write_string(string.format("pingpong %d rounds, %s us a round\n",
            [i(Rounds), s(us(T1 - T0, Rounds))]), !IO)
    else if
        [ "spawn", Text | Rest ] = Arguments,
        string.to_int(Text, Count),
        ( Rest = [], Bytes = 0
        ; Rest = [B], string.to_int(B, Bytes)
        )
    then
        new_mailbox(Done, !IO),
        new_mailbox(Pause, !IO),
        resident_kib(K0, !IO),
        now(T0, !IO),
        spawn_all(Count, Bytes, Done, [], Boxes, !IO),
        now(T1, !IO),
        yield(Pause, !IO),
        now(T2, !IO),
        resident_kib(K1, !IO),
        tell_all(Boxes, !IO),
        collect(Count, Done, !IO),
        now(T3, !IO),
        io.write_string(string.format(
            "spawn %d: %s us each; first run %s us each; message and end %s us each; %d KiB each\n",
            [i(Count), s(us(T1 - T0, Count)), s(us(T2 - T1 - 1000000, Count)),
             s(us(T3 - T2, Count)), i((K1 - K0) / Count)]), !IO)
    else
        io.write_string("usage: processes pingpong ROUNDS | work COUNT SIZE | pairs COUNT ROUNDS | spawn COUNT [BYTES]\n", !IO),
        io.set_exit_status(1, !IO)
    ).
