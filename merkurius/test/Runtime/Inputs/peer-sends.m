% Direct sends between the GPU processes of two GPUs (runtime/device/
% device.cpp, Direct sends between the GPUs of a node): the order of words,
% terms and arrays; a sink with sources on both GPUs; a source that ends
% before its sink takes its arrays; arrays to a sink that has ended, far
% more than a send pool holds; a sink that ends with arrays queued and its
% source waiting on a full queue; and a flood after them all. The argument
% is the arrays' words. Sources report to `Sent`, sinks to `Took`.
:- module peer_sends.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, process, string.

:- type rec ---> rec(int, int, float, pair) ; other(int).
:- type pair ---> pair(int, int).

:- func rec(int) = rec.
rec(I) = rec(I, I * 3, float(I) / 2.0, pair(I + 1, I * 7)).

    % Receives Count words, and reports how many came in order from 1.
:- process wsink(mailbox(int), int) receives int on gpu.
wsink(Took, Count, Self, !IO) :-
    wtake(Self, Count, 1, 0, Right, !IO),
    send(Took, Right, !IO).

:- pred wtake(mailbox(int)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
wtake(Self, Count, Next, !Right, !IO) :-
    ( if Count > 0 then
        receive(Self, any, after(60000), R, !IO),
        ( R = message(N),
            ( if N = Next then !:Right = !.Right + 1 else true ),
            wtake(Self, Count - 1, N + 1, !Right, !IO)
        ; R = timeout )
    else true ).

:- process wsource(mailbox(int), mailbox(int), int) receives int on gpu.
wsource(To, Sent, N, _Self, !IO) :-
    wput(To, 1, N, !IO),
    send(Sent, N, !IO).

:- pred wput(mailbox(int)::in, int::in, int::in, io::di, io::uo) is det.
wput(To, I, N, !IO) :-
    ( if I =< N then send(To, I, !IO), wput(To, I + 1, N, !IO) else true ).

    % Receives Count terms, and reports how many were rec(I) in order from 1.
:- process tsink(mailbox(int), int) receives rec on gpu.
tsink(Took, Count, Self, !IO) :-
    ttake(Self, Count, 1, 0, Right, !IO),
    send(Took, Right, !IO).

:- pred ttake(mailbox(rec)::in, int::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
ttake(Self, Count, Next, !Right, !IO) :-
    ( if Count > 0 then
        receive(Self, any, after(60000), R, !IO),
        ( R = message(M),
            ( if M = rec(Next) then !:Right = !.Right + 1 else true ),
            ( if M = rec(I, _, _, _) then After = I + 1 else After = Next ),
            ttake(Self, Count - 1, After, !Right, !IO)
        ; R = timeout )
    else true ).

:- process tsource(mailbox(rec), mailbox(int), int) receives int on gpu.
tsource(To, Sent, N, _Self, !IO) :-
    tput(To, 1, N, !IO),
    send(Sent, N, !IO).

:- pred tput(mailbox(rec)::in, int::in, int::in, io::di, io::uo) is det.
tput(To, I, N, !IO) :-
    ( if I =< N then send(To, rec(I), !IO), tput(To, I + 1, N, !IO) else true ).

    % Receives Count arrays, each of Words words, with its source's tag, 0
    % or 1, in element 0, its number from 1 in element 1, and Words last;
    % and reports how many were so and in order of their source.
:- process sink(mailbox(int), int, int) receives array(int) on gpu.
sink(Took, Count, Words, Self, !IO) :-
    take(Self, Count, Words, 1, 1, 0, Right, !IO),
    send(Took, Right, !IO).

:- pred take(mailbox(array(int))::in, int::in, int::in, int::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
take(Self, Count, Words, Next0, Next1, !Right, !IO) :-
    ( if Count > 0 then
        receive(Self, any, after(60000), R, !IO),
        ( R = message(A),
            ( if
                array.size(A) = Words,
                A ^ elem(Words - 1) = Words,
                Tag = A ^ elem(0),
                I = A ^ elem(1),
                ( Tag = 0, I = Next0 ; Tag = 1, I = Next1 )
            then
                !:Right = !.Right + 1,
                ( if Tag = 0 then
                    take(Self, Count - 1, Words, I + 1, Next1, !Right, !IO)
                else
                    take(Self, Count - 1, Words, Next0, I + 1, !Right, !IO)
                )
            else
                take(Self, Count - 1, Words, Next0, Next1, !Right, !IO)
            )
        ; R = timeout )
    else true ).

:- process source(mailbox(array(int)), mailbox(int), int, int, int)
    receives int on gpu.
source(To, Sent, Tag, N, Words, _Self, !IO) :-
    put(To, Tag, 1, N, Words, !IO),
    send(Sent, N, !IO).

:- pred put(mailbox(array(int))::in, int::in, int::in, int::in, int::in,
    io::di, io::uo) is det.
put(To, Tag, I, N, Words, !IO) :-
    ( if I =< N then
        A0 = array.init(Words, Words),
        A1 = A0 ^ elem(0) := Tag,
        A = A1 ^ elem(1) := I,
        send(To, A, !IO),
        put(To, Tag, I + 1, N, Words, !IO)
    else true ).

:- pred wait(mailbox(int)::in, string::in, io::di, io::uo) is det.
wait(Box, What, !IO) :-
    receive(Box, any, after(120000), R, !IO),
    ( R = message(N), io.format("%s: %d\n", [s(What), i(N)], !IO)
    ; R = timeout, io.format("%s: timeout\n", [s(What)], !IO) ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = [W], string.to_int(W, W0) then Words = W0 else Words = 4096 ),
    new_mailbox(Sent, !IO),
    new_mailbox(Took, !IO),
    % From GPU 1 to GPU 0, in order.
    spawn(gpu(0), wsink(Took, 2000), WSink, !IO),
    spawn(gpu(1), wsource(WSink, Sent, 2000), _, !IO),
    wait(Sent, "words sent", !IO),
    wait(Took, "words in order", !IO),
    spawn(gpu(0), tsink(Took, 1000), TSink, !IO),
    spawn(gpu(1), tsource(TSink, Sent, 1000), _, !IO),
    wait(Sent, "terms sent", !IO),
    wait(Took, "terms in order", !IO),
    % One source on each GPU, to a sink on GPU 1.
    spawn(gpu(1), sink(Took, 1000, Words), SinkE, !IO),
    spawn(gpu(1), source(SinkE, Sent, 0, 500, Words), _, !IO),
    spawn(gpu(0), source(SinkE, Sent, 1, 500, Words), _, !IO),
    wait(Sent, "a source sent", !IO),
    wait(Sent, "a source sent", !IO),
    wait(Took, "arrays from both GPUs in order", !IO),
    % The source ends before the sink takes its arrays: the sink takes them
    % after the host's array, which is not one of them.
    spawn(gpu(1), sink(Took, 51, Words), SinkA, !IO),
    spawn(gpu(0), source(SinkA, Sent, 0, 50, Words), _, !IO),
    wait(Sent, "an ending source sent", !IO),
    sleep_ms(200, !IO),
    send(SinkA, array.init(1, 1), !IO),
    wait(Took, "its sink took in order", !IO),
    % The sink has ended before the source sends to it.
    spawn(gpu(1), sink(Took, 1, Words), SinkB, !IO),
    send(SinkB, array.init(1, 1), !IO),
    wait(Took, "a sink took and ended", !IO),
    sleep_ms(200, !IO),
    spawn(gpu(0), source(SinkB, Sent, 0, 3000, Words), _, !IO),
    wait(Sent, "a source sent to it", !IO),
    % The sink ends with arrays queued, and its source waiting.
    spawn(gpu(1), sink(Took, 5, Words), SinkC, !IO),
    spawn(gpu(0), source(SinkC, Sent, 0, 500, Words), _, !IO),
    wait(Took, "a sink took 5 and ended", !IO),
    wait(Sent, "its source sent", !IO),
    % After them all, a flood.
    spawn(gpu(1), sink(Took, 3000, Words), SinkD, !IO),
    spawn(gpu(0), source(SinkD, Sent, 0, 3000, Words), _, !IO),
    wait(Sent, "a flood sent", !IO),
    wait(Took, "a flood taken in order", !IO).

:- pragma foreign_decl("C", "#include <time.h>").
:- pred sleep_ms(int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C", sleep_ms(Ms::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"struct timespec ts = {Ms / 1000, (Ms % 1000) * 1000000L};
nanosleep(&ts, 0);").
