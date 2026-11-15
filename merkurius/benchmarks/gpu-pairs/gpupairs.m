% Arrays between GPU processes on two GPUs of one node, through relays: a
% relay is a GPU process that forwards each array it receives to the
% mailbox it was given. The host sends arrays into a chain of relays and
% receives them at its end:
%
%   A      host -> relay on GPU 0 -> host
%   A-B    host -> relay on GPU 0 -> relay on GPU 1 -> host
%   A-A    host -> relay on GPU 0 -> relay on GPU 0 -> host
%   AB*4   host -> 8 relays, on GPU 0 and GPU 1 in turn -> host
%   AA*4   host -> 8 relays on GPU 0 -> host
%
% so A-B less A is about one hop from a GPU to another, and A-A less A one
% hop within a GPU, between processes of one kernel; AB*4 less A is 7 hops
% between the GPUs, and AA*4 less A 7 within one. In `ping` one array is
% in flight, for latency; in `stream` all of a measurement's are, for
% bandwidth. Each measurement moves MIB mebibytes (the argument, 64 by
% default), in 2 to 20000 arrays, 2000 at most for `ping`. Run with two
% GPUs visible, as utils/gpu/with-gpus.py --gpus A,B gives; with one, A-B
% and AB*4 are left out (README.md).

:- module gpupairs.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, process, string.

    % Forwards each array to To, and ends after an empty one, which it
    % forwards too.
:- process relay(mailbox(array(int))) receives array(int) on gpu.

relay(To, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(A),
        Size = array.size(A),
        send(To, A, !IO),
        ( if Size > 0 then
            relay(To, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

    % As relay, but a kind with a loop, so each relay has worker teams,
    % which copy its arrays into and out of messages (runtime/device,
    % Bulk). The loop sums the array, and runs only for an array whose
    % first element is negative, which the benchmark never sends.
:- process wrelay(mailbox(array(int))) receives array(int) on gpu.

wrelay(To, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(A),
        Size = array.size(A),
        ( if Size > 0, A ^ elem(0) < 0 then
            %$omp distribute parallel do reduction((+) : Sum)
            ( for(I, 0, Size - 1), fromto(0, S0, S1, Sum) do
                S1 = S0 + A ^ elem(I)
            ),
            send(To, array.init(1, Sum), !IO)
        else
            send(To, A, !IO)
        ),
        ( if Size > 0 then
            wrelay(To, Self, !IO)
        else
            true
        )
    ;
        R = timeout
    ).

:- func sizes = list(int).
sizes = [1, 16, 256, 1024, 4096, 16384, 65536, 262144, 1048576].

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [M | _], string.to_int(M, Mib0) then Mib = Mib0 else Mib = 64 ),
    % "workers K": relays with worker teams, each taking up to a K'th of its
    % GPU's kernel's teams.
    ( if Args = [_, "workers", KText], string.to_int(KText, K) then
        Workers = K
    else
        Workers = 0
    ),
    gpus(Gpus, !IO),
    ( if Workers > 0 then
        set_worker_teams(gpu(0), share(Workers), !IO),
        ( if Gpus >= 2 then
            set_worker_teams(gpu(1), share(Workers), !IO)
        else
            true
        )
    else
        true
    ),
    ( if Gpus < 1 then
        io.write_string("gpupairs: no GPU\n", !IO)
    else
        new_mailbox(Back, !IO),
        chain(Workers, "A", [gpu(0)], Back, Mib, !IO),
        ( if Gpus >= 2 then
            chain(Workers, "A-B", [gpu(0), gpu(1)], Back, Mib, !IO)
        else
            true
        ),
        chain(Workers, "A-A", [gpu(0), gpu(0)], Back, Mib, !IO),
        ( if Gpus >= 2 then
            chain(Workers, "AB*4", [gpu(0), gpu(1), gpu(0), gpu(1), gpu(0),
                gpu(1), gpu(0), gpu(1)], Back, Mib, !IO)
        else
            true
        ),
        chain(Workers, "AA*4", [gpu(0), gpu(0), gpu(0), gpu(0), gpu(0),
            gpu(0), gpu(0), gpu(0)], Back, Mib, !IO)
    ).

    % Spawns the relays at Places, the last forwarding to Back, measures
    % every size both ways, and ends the relays.
:- pred chain(int::in, string::in, list(place)::in, mailbox(array(int))::in,
    int::in, io::di, io::uo) is det.
chain(Workers, Name, Places, Back, Mib, !IO) :-
    spawn_chain(Workers, list.reverse(Places), Back, First, !IO),
    % The first round trip starts the kernels.
    send(First, array.init(1, 1), !IO),
    receive(Back, any, after(60000), _, !IO),
    list.foldl(measure(Name, "ping", First, Back, Mib), sizes, !IO),
    list.foldl(measure(Name, "stream", First, Back, Mib), sizes, !IO),
    send(First, array.init(0, 0), !IO),
    receive(Back, any, after(60000), _, !IO).

:- pred spawn_chain(int::in, list(place)::in, mailbox(array(int))::in,
    mailbox(array(int))::out, io::di, io::uo) is det.
spawn_chain(_, [], To, To, !IO).
spawn_chain(Workers, [Place | Places], To, First, !IO) :-
    ( if Workers > 0 then
        spawn(Place, wrelay(To), Relay, !IO)
    else
        spawn(Place, relay(To), Relay, !IO)
    ),
    spawn_chain(Workers, Places, Relay, First, !IO).

:- pred measure(string::in, string::in, mailbox(array(int))::in,
    mailbox(array(int))::in, int::in, int::in, io::di, io::uo) is det.
measure(Name, Mode, First, Back, Mib, Words, !IO) :-
    Total = Mib * 1048576 / 8 / Words,
    Most = ( if Mode = "ping" then 2000 else 20000 ),
    Rounds = int.max(2, int.min(Most, Total)),
    A = array.init(Words, Words),
    % One untimed round trip of this size.
    send(First, A, !IO),
    receive(Back, any, after(60000), _, !IO),
    now(Start, !IO),
    ( if Mode = "ping" then
        ping(First, Back, A, Words, Rounds, 0, Right, !IO)
    else
        send_all(First, A, Rounds, !IO),
        receive_all(Back, Words, Rounds, 0, Right, !IO)
    ),
    now(End, !IO),
    report(Name, Mode, Words, Rounds, Right, End - Start, !IO).

:- pred ping(mailbox(array(int))::in, mailbox(array(int))::in,
    array(int)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
ping(First, Back, A, Words, Rounds, !Right, !IO) :-
    ( if Rounds > 0 then
        send(First, A, !IO),
        receive_all(Back, Words, 1, !Right, !IO),
        ping(First, Back, A, Words, Rounds - 1, !Right, !IO)
    else
        true
    ).

:- pred send_all(mailbox(array(int))::in, array(int)::in, int::in,
    io::di, io::uo) is det.
send_all(To, A, Rounds, !IO) :-
    ( if Rounds > 0 then
        send(To, A, !IO),
        send_all(To, A, Rounds - 1, !IO)
    else
        true
    ).

    % Receives Rounds arrays; Right counts those of the right size whose
    % last element is their size.
:- pred receive_all(mailbox(array(int))::in, int::in, int::in, int::in,
    int::out, io::di, io::uo) is det.
receive_all(From, Words, Rounds, !Right, !IO) :-
    ( if Rounds > 0 then
        receive(From, any, after(60000), Got, !IO),
        (
            Got = message(A),
            ( if array.size(A) = Words, A ^ elem(Words - 1) = Words then
                !:Right = !.Right + 1
            else
                true
            ),
            receive_all(From, Words, Rounds - 1, !Right, !IO)
        ;
            Got = timeout
        )
    else
        true
    ).

:- pragma foreign_decl("C", "
#include <stdio.h>
#include <time.h>
").

:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(Nanoseconds::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    Nanoseconds = (MR_Integer) ts.tv_sec * 1000000000 + ts.tv_nsec;
").

:- pred report(string::in, string::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
:- pragma foreign_proc("C",
    report(Chain::in, Mode::in, Words::in, Rounds::in, Right::in,
        Nanoseconds::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-5s %-6s %8ld words %6ld arrays %6ld right %10.1f MB/s %12.2f us each\\n\",
        Chain, Mode, (long) Words, (long) Rounds, (long) Right,
        8.0 * Words * Rounds * 1000.0 / Nanoseconds,
        Nanoseconds / 1000.0 / Rounds);
    fflush(stdout);
").
