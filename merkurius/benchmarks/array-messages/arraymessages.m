% The bandwidth of arrays as messages, from a GPU's target region and from
% the CPU (README.md), over message sizes of 1 to 2^20 words. In one
% direction, "to", node 1 sends the arrays to its partner; in the other,
% "from", it receives them. The partner is node 2, another process over
% TCP, or node 1's own main, before or after the region, which leaves TCP
% out. The time is taken on the host, less that of a run with no messages,
% which has the region's launch and the making of the array.
%
% Usage, one process for each node:
%   arraymessages partner-gpu           node 2, its operations in target
%                                       regions
%   arraymessages partner               node 2: listens, prints its port,
%                                       and serves until told to stop
%   arraymessages cpu PORT MIB          node 1's main, against node 2
%   arraymessages gpu PORT MIB          node 1's target region, against node 2
%   arraymessages cpu-local MIB         node 1's main, its own mailbox
%   arraymessages gpu-local MIB         node 1's target region, against main
%
% MIB is how many mebibytes each measurement moves, in at least two messages
% and at most 20000.

:- module arraymessages.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, int, list, maybe, process, string.

:- type arrays == array(int).

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["partner"] then
        partner(no, !IO)
    else if Arguments = ["partner-gpu"] then
        partner(yes, !IO)
    else if
        Arguments = [Where, Port, MiBString],
        ( Where = "cpu" ; Where = "gpu" ),
        string.to_int(MiBString, MiB)
    then
        remote(Where, Port, MiB, !IO)
    else if
        Arguments = [Where, MiBString],
        ( Where = "cpu-local" ; Where = "gpu-local" ),
        string.to_int(MiBString, MiB)
    then
        new_mailbox(Local : mailbox(arrays), !IO),
        sizes(Where, no, Local, Local, Local, MiB, 1, !IO)
    else
        io.write_string("usage: partner | cpu PORT MIB | gpu PORT MIB" ++
            " | cpu-local MIB | gpu-local MIB\n", !IO),
        io.set_exit_status(2, !IO)
    ).

%---------------------------------------------------------------------------%
% Node 2.

    % With OnGpu, each of its operations is a target region of its own, so
    % that the arena of what it receives goes at the operation's end.
:- pred partner(bool::in, io::di, io::uo) is det.
partner(OnGpu, !IO) :-
    new_mailbox(Commands : mailbox(arrays), !IO),
    new_mailbox(Data : mailbox(arrays), !IO),
    new_mailbox(Hello : mailbox(mailbox(arrays)), !IO),
    register("commands", Commands, _, !IO),
    register("data", Data, _, !IO),
    register("hello", Hello, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        receive(Hello, any, infinity, GotNode1, !IO),
        (
            GotNode1 = message(Node1),
            serve(OnGpu, Commands, Data, Node1, !IO)
        ;
            GotNode1 = timeout
        )
    ;
        Listening = error(Error),
        io.write_string("listen: " ++ Error ++ "\n", !IO),
        io.set_exit_status(1, !IO)
    ).

    % A command is [Operation, Words, Rounds]: 1 receives Rounds arrays and
    % answers with how many had Words elements, 2 sends Rounds arrays of
    % Words elements, and anything else ends.
:- pred serve(bool::in, mailbox(arrays)::in, mailbox(arrays)::in,
    mailbox(arrays)::in, io::di, io::uo) is det.
serve(OnGpu, Commands, Data, Node1, !IO) :-
    receive(Commands, any, infinity, Got, !IO),
    (
        Got = message(Command),
        Operation = Command ^ elem(0),
        Words = Command ^ elem(1),
        Rounds = Command ^ elem(2),
        ( if Operation = 1 then
            (
                OnGpu = yes,
                receive_on_gpu(Data, Words, Rounds, Right, !IO)
            ;
                OnGpu = no,
                receive_arrays(Data, Words, Rounds, 0, Right, !IO)
            ),
            send(Node1, array.init(1, Right), !IO),
            serve(OnGpu, Commands, Data, Node1, !IO)
        else if Operation = 2 then
            (
                OnGpu = yes,
                send_on_gpu(Node1, Words, Rounds, !IO)
            ;
                OnGpu = no,
                send_arrays(Node1, Words, Rounds, !IO)
            ),
            serve(OnGpu, Commands, Data, Node1, !IO)
        else
            true
        )
    ;
        Got = timeout
    ).

%---------------------------------------------------------------------------%
% What both sides do, on the CPU or in a target region.

:- pred send_arrays(mailbox(arrays)::in, int::in, int::in,
    io::di, io::uo) is det.
send_arrays(To, Words, Rounds, !IO) :-
    A = array.init(Words, Words),
    send_times(To, A, Rounds, !IO).

:- pred send_times(mailbox(arrays)::in, arrays::in, int::in,
    io::di, io::uo) is det.
send_times(To, A, Rounds, !IO) :-
    ( if Rounds =< 0 then
        true
    else
        send(To, A, !IO),
        send_times(To, A, Rounds - 1, !IO)
    ).

    % Receives Rounds arrays; Right counts those of the right size whose
    % last element is their size.
:- pred receive_arrays(mailbox(arrays)::in, int::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
receive_arrays(From, Words, Rounds, !Right, !IO) :-
    ( if Rounds =< 0 then
        true
    else
        receive(From, any, after(60000), Got, !IO),
        (
            Got = message(A),
            ( if
                array.size(A) = Words,
                A ^ elem(Words - 1) = Words
            then
                !:Right = !.Right + 1
            else
                true
            ),
            receive_arrays(From, Words, Rounds - 1, !Right, !IO)
        ;
            Got = timeout
        )
    ).

:- pred send_on_gpu(mailbox(arrays)::in, int::in, int::in,
    io::di, io::uo) is det.
send_on_gpu(To, Words, Rounds, !IO) :-
    %$omp target
    send_arrays(To, Words, Rounds, !IO).

:- pred receive_on_gpu(mailbox(arrays)::in, int::in, int::in, int::out,
    io::di, io::uo) is det.
receive_on_gpu(From, Words, Rounds, Right, !IO) :-
    %$omp target
    receive_arrays(From, Words, Rounds, 0, Right, !IO).

%---------------------------------------------------------------------------%
% Node 1.

:- pred remote(string::in, string::in, int::in, io::di, io::uo) is det.
remote(Where, Port, MiB, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Node),
        lookup(Node, "commands", after(10000), FoundCommands, !IO),
        lookup(Node, "data", after(10000), FoundData, !IO),
        lookup(Node, "hello", after(10000), FoundHello, !IO),
        ( if
            FoundCommands = ok(Commands),
            FoundData = ok(Data),
            FoundHello = ok(Hello)
        then
            new_mailbox(Mine : mailbox(arrays), !IO),
            send(Hello, Mine, !IO),
            sizes(Where, yes, Commands, Data, Mine, MiB, 1, !IO),
            send(Commands, array.init(3, 0), !IO)
        else
            io.write_string("lookups failed\n", !IO),
            io.set_exit_status(1, !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO),
        io.set_exit_status(1, !IO)
    ).

    % Measures each size, 1, 4, 16 and so on to 2^20 words.
:- pred sizes(string::in, bool::in, mailbox(arrays)::in, mailbox(arrays)::in,
    mailbox(arrays)::in, int::in, int::in, io::di, io::uo) is det.
sizes(Where, Remote, Commands, To, From, MiB, Words, !IO) :-
    ( if Words > 1048576 then
        true
    else
        Rounds = int.max(2, int.min(20000, MiB * 1048576 / (8 * Words))),
        measure_to(Where, Remote, Commands, To, From, Words, Rounds, !IO),
        measure_from(Where, Remote, Commands, To, From, Words, Rounds, !IO),
        sizes(Where, Remote, Commands, To, From, MiB, 4 * Words, !IO)
    ).

:- pred is_gpu(string::in) is semidet.
is_gpu("gpu").
is_gpu("gpu-local").

:- pred sending(string::in, mailbox(arrays)::in, int::in, int::in,
    int::out, io::di, io::uo) is det.
sending(Where, To, Words, Rounds, Nanoseconds, !IO) :-
    now(T0, !IO),
    ( if is_gpu(Where) then
        send_on_gpu(To, Words, Rounds, !IO)
    else
        send_arrays(To, Words, Rounds, !IO)
    ),
    now(T1, !IO),
    Nanoseconds = T1 - T0.

:- pred receiving(string::in, mailbox(arrays)::in, int::in, int::in,
    int::out, int::out, io::di, io::uo) is det.
receiving(Where, From, Words, Rounds, Right, Nanoseconds, !IO) :-
    now(T0, !IO),
    ( if is_gpu(Where) then
        receive_on_gpu(From, Words, Rounds, Right, !IO)
    else
        receive_arrays(From, Words, Rounds, 0, Right, !IO)
    ),
    now(T1, !IO),
    Nanoseconds = T1 - T0.

    % Node 1 sends. Node 2 counts them and answers, and the time includes
    % its answer; in node 1, main receives them after the time.
:- pred measure_to(string::in, bool::in, mailbox(arrays)::in,
    mailbox(arrays)::in, mailbox(arrays)::in, int::in, int::in,
    io::di, io::uo) is det.
measure_to(Where, Remote, Commands, To, From, Words, Rounds, !IO) :-
    sending(Where, To, Words, 0, Empty, !IO),
    (
        Remote = yes,
        now(T0, !IO),
        send(Commands, array.from_list([1, Words, Rounds]), !IO),
        sending(Where, To, Words, Rounds, _, !IO),
        receive(From, any, after(60000), Answer, !IO),
        now(T1, !IO),
        Time = T1 - T0,
        ( if Answer = message(A) then Right = A ^ elem(0) else Right = -1 )
    ;
        Remote = no,
        sending(Where, To, Words, Rounds, Time, !IO),
        receive_arrays(From, Words, Rounds, 0, Right, !IO)
    ),
    report(Where, "to", Words, Rounds, Right, int.max(1, Time - Empty), !IO).

    % Node 1 receives. Node 2 starts on the command; in node 1, main sends
    % them before the time.
:- pred measure_from(string::in, bool::in, mailbox(arrays)::in,
    mailbox(arrays)::in, mailbox(arrays)::in, int::in, int::in,
    io::di, io::uo) is det.
measure_from(Where, Remote, Commands, To, From, Words, Rounds, !IO) :-
    receiving(Where, From, Words, 0, _, Empty, !IO),
    (
        Remote = yes,
        send(Commands, array.from_list([2, Words, Rounds]), !IO)
    ;
        Remote = no,
        send_arrays(To, Words, Rounds, !IO)
    ),
    receiving(Where, From, Words, Rounds, Right, Time, !IO),
    report(Where, "from", Words, Rounds, Right, int.max(1, Time - Empty), !IO).

%---------------------------------------------------------------------------%

:- pragma foreign_decl("C", local, "
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
    report(Where::in, Direction::in, Words::in, Rounds::in, Right::in,
        Nanoseconds::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-10s %-5s %8ld words %6ld messages %6ld right %10.1f MB/s %12.2f us each\\n\",
        Where, Direction, (long) Words, (long) Rounds, (long) Right,
        8.0 * Words * Rounds * 1000.0 / Nanoseconds,
        Nanoseconds / 1000.0 / Rounds);
    fflush(stdout);
").
