% Arrays from GPU processes on one GPU straight to GPU processes on another,
% with no host on the way: what the direct path between GPUs gives, against
% the link (README.md). A source makes an array of WORDS words once and
% sends it COUNT times to its sink, which checks each and times them on its
% GPU's clock, from the first array it receives to the last:
%
%   A>A      1 source and its sink on GPU 0
%   A>B      1 source on GPU 0, its sink on GPU 1
%   A>Bx4    4 such pairs at once
%   A<>B     1 pair each way at once
%   A<>Bx4   4 pairs each way at once
%
% Each measurement moves MIB mebibytes (the argument, 256 by default), split
% between the pairs. The host spawns the pairs, waits until each source's
% first message, of 1 word, has reached its sink, so that the kernels run
% and the routes are made, and then tells the sources to start. "device
% MB/s" is, for each GPU with sinks, the arrays after each sink's first
% over the span on that GPU's clock from the earliest first to the latest
% last, summed over the GPUs; it leaves out the host and the first array's
% latency. "host MB/s" is all the arrays over the host's time from the
% start to the last sink's report. "us each" is the mean time between
% arrays at a sink. Run with two GPUs visible, as
% utils/gpu/with-gpus.py --gpus A,B gives; with one, only A>A runs.
% "workers K" gives each source and sink worker teams, up to a K'th of its
% GPU's kernel's teams, which copy its arrays (run.sh builds it with
% -fopenmp then).

:- module link.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

:- type pair_at
    --->    pair_at(place, place).    % The source's place, the sink's.

    % Sends its sink 1 word, waits for the host's word, and sends the sink
    % Words words Count times.
:- process source(int, int, mailbox(array(int))) receives int on gpu.

source(Words, Count, To, Self, !IO) :-
    source_body(Words, Count, To, Self, !IO).

:- pred source_body(int::in, int::in, mailbox(array(int))::in,
    mailbox(int)::in, io::di, io::uo) is det.

source_body(Words, Count, To, Self, !IO) :-
    A = array.init(Words, Words),
    send(To, array.init(1, 0), !IO),
    receive(Self, any, infinity, _, !IO),
    send_n(To, A, Count, !IO).

:- pred send_n(mailbox(array(int))::in, array(int)::in, int::in,
    io::di, io::uo) is det.

send_n(To, A, Count, !IO) :-
    ( if Count > 0 then
        send(To, A, !IO),
        send_n(To, A, Count - 1, !IO)
    else
        true
    ).

    % Takes the source's word and tells Report (-1); then takes Count
    % arrays, and tells Report how many were right, and the times on its
    % GPU's clock at the first and the last.
:- process sink(int, int, mailbox(int)) receives array(int) on gpu.

sink(Words, Count, Report, Self, !IO) :-
    sink_body(Words, Count, Report, Self, !IO).

:- pred sink_body(int::in, int::in, mailbox(int)::in,
    mailbox(array(int))::in, io::di, io::uo) is det.

sink_body(Words, Count, Report, Self, !IO) :-
    receive(Self, any, infinity, _, !IO),
    send(Report, -1, !IO),
    take(Self, Words, 1, 0, First, !IO),
    now(Start, !IO),
    take(Self, Words, Count - 1, First, Right, !IO),
    now(End, !IO),
    send(Report, Right, !IO),
    send(Report, Start, !IO),
    send(Report, End, !IO).

    % Takes Count arrays; Right counts those of Words words whose last
    % element is Words.
:- pred take(mailbox(array(int))::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.

take(Self, Words, Count, !Right, !IO) :-
    ( if Count > 0 then
        receive(Self, any, infinity, R, !IO),
        ( if
            R = message(A),
            array.size(A) = Words,
            A ^ elem(Words - 1) = Words
        then
            !:Right = !.Right + 1
        else
            true
        ),
        take(Self, Words, Count - 1, !Right, !IO)
    else
        true
    ).

    % As source and sink, but kinds with a loop, so each has worker teams,
    % which copy its arrays into and out of messages (runtime/device, Bulk).
    % The loop runs only for a negative Words, which the benchmark never
    % gives.
:- process wsource(int, int, mailbox(array(int))) receives int on gpu.

wsource(Words, Count, To, Self, !IO) :-
    ( if Words < 0 then
        A = array.init(Count, Words),
        %$omp distribute parallel do reduction((+) : Sum)
        ( for(I, 0, Count - 1), fromto(0, S0, S1, Sum) do
            S1 = S0 + A ^ elem(I)
        ),
        send(To, array.init(1, Sum), !IO)
    else
        source_body(Words, Count, To, Self, !IO)
    ).

:- process wsink(int, int, mailbox(int)) receives array(int) on gpu.

wsink(Words, Count, Report, Self, !IO) :-
    ( if Words < 0 then
        A = array.init(Count, Words),
        %$omp distribute parallel do reduction((+) : Sum)
        ( for(I, 0, Count - 1), fromto(0, S0, S1, Sum) do
            S1 = S0 + A ^ elem(I)
        ),
        send(Report, Sum, !IO)
    else
        sink_body(Words, Count, Report, Self, !IO)
    ).

:- func sizes = list(int).
sizes = [1, 1024, 4096, 16384, 65536, 262144, 1048576].

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [M | _], string.to_int(M, Mib0) then
        Mib = Mib0
    else
        Mib = 256
    ),
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
    A = gpu(0),
    B = gpu(1),
    ( if Gpus < 1 then
        io.write_string("link: no GPU\n", !IO)
    else
        case(Workers, Mib, "A>A", [pair_at(A, A)], !IO),
        ( if Gpus >= 2 then
            case(Workers, Mib, "A>B", [pair_at(A, B)], !IO),
            case(Workers, Mib, "A>Bx4", list.duplicate(4, pair_at(A, B)),
                !IO),
            case(Workers, Mib, "A<>B", [pair_at(A, B), pair_at(B, A)], !IO),
            case(Workers, Mib, "A<>Bx4",
                list.duplicate(4, pair_at(A, B)) ++
                list.duplicate(4, pair_at(B, A)), !IO)
        else
            true
        )
    ).

:- pred case(int::in, int::in, string::in, list(pair_at)::in,
    io::di, io::uo) is det.

case(Workers, Mib, Name, Pairs, !IO) :-
    list.foldl(measure(Workers, Mib, Name, Pairs), sizes, !IO).

:- pred measure(int::in, int::in, string::in, list(pair_at)::in, int::in,
    io::di, io::uo) is det.

measure(Workers, Mib, Name, Pairs, Words, !IO) :-
    N = list.length(Pairs),
    Count = int.max(2, int.min(20000, Mib * 1048576 / 8 / Words / N)),
    list.map_foldl(start_pair(Workers, Words, Count), Pairs, Started, !IO),
    now(Start, !IO),
    list.foldl(go, Started, !IO),
    list.map_foldl(finish, Started, Ended, !IO),
    now(End, !IO),
    Right = list.foldl((func(ended(_, R, _, _), T) = T + R), Ended, 0),
    Each = 8 * Words * (Count - 1),
    Rate = list.foldl(
        (func(Place, T) = T + gpu_rate(Each, Place, Ended)),
        list.sort_and_remove_dups(list.map((func(ended(P, _, _, _)) = P),
            Ended)), 0.0),
    Spans = list.foldl((func(ended(_, _, S, E), T) = T + (E - S)), Ended, 0),
    UsEach = float(Spans) / 1000.0 / float(N * (Count - 1)),
    Bytes = 8 * Words * Count * N,
    report(Name, Words, Count * N, Right, Rate,
        float(Bytes) * 1000.0 / float(End - Start), UsEach, !IO).

    % The MB/s into the sinks at Place: their arrays after their first, over
    % the span from the earliest first to the latest last on its clock.
:- func gpu_rate(int, place, list(ended)) = float.

gpu_rate(Each, Place, Ended) = Rate :-
    At = list.filter((pred(ended(P, _, _, _)::in) is semidet :- P = Place),
        Ended),
    First = list.foldl((func(ended(_, _, S, _), M) = int.min(M, S)), At,
        int.max_int),
    Last = list.foldl((func(ended(_, _, _, E), M) = int.max(M, E)), At,
        int.min_int),
    Rate = ( if Last > First then
        float(Each * list.length(At)) * 1000.0 / float(Last - First)
    else
        0.0
    ).

:- type started
    --->    started(mailbox(int), mailbox(int), place).
            % The source, the sink's report, and the sink's place.

:- type ended
    --->    ended(place, int, int, int).
            % The sink's place, its right arrays, and its first and last.

:- pred start_pair(int::in, int::in, int::in, pair_at::in, started::out,
    io::di, io::uo) is det.

start_pair(Workers, Words, Count, pair_at(From, To), Started, !IO) :-
    new_mailbox(Report, !IO),
    ( if Workers > 0 then
        spawn(To, wsink(Words, Count, Report), Sink, !IO),
        spawn(From, wsource(Words, Count, Sink), Source, !IO)
    else
        spawn(To, sink(Words, Count, Report), Sink, !IO),
        spawn(From, source(Words, Count, Sink), Source, !IO)
    ),
    receive(Report, any, after(60000), _, !IO),
    Started = started(Source, Report, To).

:- pred go(started::in, io::di, io::uo) is det.

go(started(Source, _, _), !IO) :-
    send(Source, 0, !IO).

    % A sink's report.
:- pred finish(started::in, ended::out, io::di, io::uo) is det.

finish(started(_, Report, Place), Ended, !IO) :-
    receive(Report, any, after(60000), R1, !IO),
    receive(Report, any, after(60000), R2, !IO),
    receive(Report, any, after(60000), R3, !IO),
    ( if R1 = message(Right), R2 = message(S), R3 = message(E) then
        Ended = ended(Place, Right, S, E)
    else
        Ended = ended(Place, 0, 0, 0)
    ).

:- pragma foreign_decl("C", "
#include <stdio.h>
#include <time.h>
").

    % Nanoseconds by the steady clock: the host's, or a GPU's in a GPU
    % process (runtime/device, Now).
:- pred now(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(Nanoseconds::out, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure, thread_safe],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    Nanoseconds = (MR_Integer) ts.tv_sec * 1000000000 + ts.tv_nsec;
").

:- pred report(string::in, int::in, int::in, int::in, float::in, float::in,
    float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    report(Case::in, Words::in, Arrays::in, Right::in, DeviceRate::in,
        HostRate::in, UsEach::in, _IO0::di, _IO::uo),
    [will_not_call_merkurius, promise_pure],
"
    printf(\"%-7s %8ld words %6ld arrays %6ld right %10.1f device MB/s %10.1f host MB/s %10.2f us each\\n\",
        Case, (long) Words, (long) Arrays, (long) Right, DeviceRate,
        HostRate, UsEach);
    fflush(stdout);
").
