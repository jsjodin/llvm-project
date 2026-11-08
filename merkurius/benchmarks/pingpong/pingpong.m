% Ping-pong latency between processes, from a GPU's target region and from
% the CPU (README.md). Each round, node 1 sends N and waits for N + 1. Its
% partner is node 2, another process over TCP, or an echo thread in node 1
% itself, which leaves TCP out. The rounds are timed on the host with
% clock_gettime, after warm-up rounds, so that a target region's time
% includes its launch, which an empty region measures.
%
% Usage, one process for each node:
%   pingpong partner                 node 2: listens, prints its port,
%                                    and answers until -1
%   pingpong cpu PORT ROUNDS         node 1's main, against node 2
%   pingpong gpu PORT ROUNDS         node 1's target region, against node 2
%   pingpong cpu-local ROUNDS        node 1's main, against its echo thread
%   pingpong gpu-local ROUNDS        node 1's target region, against its
%                                    echo thread

:- module pingpong.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- func warm_up = int.
warm_up = 100.

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["partner"] then
        partner(!IO)
    else if
        Arguments = [Where, Port, RoundsString],
        ( Where = "cpu" ; Where = "gpu" ),
        string.to_int(RoundsString, Rounds)
    then
        remote(Where, Port, Rounds, !IO)
    else if
        Arguments = [Where, RoundsString],
        ( Where = "cpu-local" ; Where = "gpu-local" ),
        string.to_int(RoundsString, Rounds)
    then
        local(Where, Rounds, !IO)
    else
        io.write_string("usage: partner | cpu PORT ROUNDS | gpu PORT ROUNDS" ++
            " | cpu-local ROUNDS | gpu-local ROUNDS\n", !IO),
        io.set_exit_status(2, !IO)
    ).

%---------------------------------------------------------------------------%
% Node 2.

:- pred partner(io::di, io::uo) is det.
partner(!IO) :-
    new_mailbox(Partner : mailbox(int), !IO),
    new_mailbox(Hello : mailbox(mailbox(int)), !IO),
    register("partner", Partner, _, !IO),
    register("hello", Hello, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        receive(Hello, any, infinity, GotNode1, !IO),
        (
            GotNode1 = yes(Node1),
            answer(Partner, Node1, !IO)
        ;
            GotNode1 = no
        )
    ;
        Listening = error(Error),
        io.write_string("listen: " ++ Error ++ "\n", !IO),
        io.set_exit_status(1, !IO)
    ).

    % Answers each N with N + 1 until -1.
:- pred answer(mailbox(int)::in, mailbox(int)::in, io::di, io::uo) is det.
answer(Partner, Node1, !IO) :-
    receive(Partner, any, infinity, Got, !IO),
    ( if Got = yes(N), N >= 0 then
        send(Node1, N + 1, !IO),
        answer(Partner, Node1, !IO)
    else
        true
    ).

%---------------------------------------------------------------------------%
% Node 1.

:- pred remote(string::in, string::in, int::in, io::di, io::uo) is det.
remote(Where, Port, Rounds, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Node),
        lookup(Node, "partner", after(10000), FoundPartner, !IO),
        lookup(Node, "hello", after(10000), FoundHello, !IO),
        ( if
            FoundPartner = ok(Partner),
            FoundHello = ok(Hello)
        then
            new_mailbox(Mine : mailbox(int), !IO),
            send(Hello, Mine, !IO),
            measure(Where, Partner, Mine, Rounds, !IO),
            send(Partner, -1, !IO)
        else
            io.write_string("lookups failed\n", !IO),
            io.set_exit_status(1, !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO),
        io.set_exit_status(1, !IO)
    ).

:- pred local(string::in, int::in, io::di, io::uo) is det.
local(Where, Rounds, !IO) :-
    new_mailbox(Echo : mailbox(int), !IO),
    new_mailbox(Mine : mailbox(int), !IO),
    start_echo(Echo, Mine, !IO),
    measure(Where, Echo, Mine, Rounds, !IO),
    send(Echo, -1, !IO),
    join_echo(!IO).

    % Warms up, then times the rounds, from main or from a target region.
:- pred measure(string::in, mailbox(int)::in, mailbox(int)::in, int::in,
    io::di, io::uo) is det.
measure(Where, Partner, Mine, Rounds, !IO) :-
    ( if ( Where = "gpu" ; Where = "gpu-local" ) then
        on_gpu(Partner, Mine, 0, warm_up, _, !IO),
        empty_region(!IO),
        now(Empty0, !IO),
        empty_region(!IO),
        now(Empty1, !IO),
        now(T0, !IO),
        on_gpu(Partner, Mine, 2 * warm_up, Rounds, InOrder, !IO),
        now(T1, !IO),
        report(Where, Rounds, InOrder, T1 - T0, !IO),
        report_launch(Empty1 - Empty0, !IO)
    else
        exchange(Partner, Mine, 0, warm_up, 0, _, !IO),
        now(T0, !IO),
        exchange(Partner, Mine, 2 * warm_up, Rounds, 0, InOrder, !IO),
        now(T1, !IO),
        report(Where, Rounds, InOrder, T1 - T0, !IO)
    ).

:- pred on_gpu(mailbox(int)::in, mailbox(int)::in, int::in, int::in,
    int::out, io::di, io::uo) is det.
on_gpu(Partner, Mine, First, Rounds, InOrder, !IO) :-
    %$omp target
    exchange(Partner, Mine, First, Rounds, 0, InOrder, !IO).

:- pred empty_region(io::di, io::uo) is det.
empty_region(!IO) :-
    One = 1,
    %$omp target
    Two = One + 1,
    ( if Two = 2 then true else io.write_string("", !IO) ).

    % Rounds rounds from N: sends N, expects N + 1, then sends N + 2.
:- pred exchange(mailbox(int)::in, mailbox(int)::in, int::in, int::in,
    int::in, int::out, io::di, io::uo) is det.
exchange(Partner, Mine, N, Rounds, !InOrder, !IO) :-
    ( if Rounds =< 0 then
        true
    else
        send(Partner, N, !IO),
        receive(Mine, any, after(10000), Got, !IO),
        (
            Got = yes(M),
            ( if M = N + 1 then
                !:InOrder = !.InOrder + 1
            else
                true
            ),
            exchange(Partner, Mine, M + 1, Rounds - 1, !InOrder, !IO)
        ;
            Got = no
        )
    ).

%---------------------------------------------------------------------------%
% What C does for the program: the clock, printing, and the echo thread.

:- pragma foreign_decl("C", local, "
#include <pthread.h>
#include <stdio.h>
#include <time.h>

extern MR_Integer merk_mailbox_wait(MR_Integer, MR_Integer, MR_Integer,
    MR_Integer, MR_Integer, MR_Integer position, MR_Integer deadline);
extern MR_Word merk_mailbox_message(MR_Integer, MR_Integer, MR_Integer,
    MR_Integer, MR_Integer, MR_Integer position);
extern void merk_mailbox_remove(MR_Integer, MR_Integer, MR_Integer,
    MR_Integer, MR_Integer, MR_Integer position);
extern void merk_mailbox_send(MR_Integer, MR_Integer, MR_Integer,
    MR_Integer, MR_Integer, MR_Word type_info, MR_Word message);

/* A mailbox handle is a cell of five words (library/process.m). */
typedef struct {
    MR_Word *in, *out;
    MR_Word type_info;
} echo_args;

static pthread_t echo_thread;
static echo_args echo;

/* Answers each N with N + 1 until -1, as node 2 does. */
static void *run_echo(void *raw)
{
    echo_args *e = raw;
    MR_Word *i = e->in, *o = e->out;
    for (;;) {
        if (!merk_mailbox_wait(i[0], i[1], i[2], i[3], i[4], 0, -1)) {
            break;
        }
        MR_Integer n = (MR_Integer)
            merk_mailbox_message(i[0], i[1], i[2], i[3], i[4], 0);
        merk_mailbox_remove(i[0], i[1], i[2], i[3], i[4], 0);
        if (n < 0) {
            break;
        }
        merk_mailbox_send(o[0], o[1], o[2], o[3], o[4], e->type_info,
            (MR_Word) (n + 1));
    }
    return NULL;
}
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

:- pred report(string::in, int::in, int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    report(Where::in, Rounds::in, InOrder::in, Nanoseconds::in,
        _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-10s %8ld rounds, %8ld in order, %10.2f us a round\\n\",
        Where, (long) Rounds, (long) InOrder,
        Rounds > 0 ? Nanoseconds / 1000.0 / Rounds : 0.0);
    fflush(stdout);
").

:- pred report_launch(int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    report_launch(Nanoseconds::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"%-10s %10.2f us\\n\", \"launch\", Nanoseconds / 1000.0);
    fflush(stdout);
").

    % Starts the echo thread, which answers In's messages to Out. T is int;
    % the predicate is polymorphic for T's type_info.
:- pred start_echo(mailbox(T)::in, mailbox(T)::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    start_echo(In::in, Out::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    echo.in = (MR_Word *) In;
    echo.out = (MR_Word *) Out;
    echo.type_info = TypeInfo_for_T;
    pthread_create(&echo_thread, NULL, run_echo, &echo);
").

:- pred join_echo(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    join_echo(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    pthread_join(echo_thread, NULL);
").
