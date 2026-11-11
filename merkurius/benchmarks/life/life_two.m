% Conway's Game of Life of life.m on two nodes, over TCP: step 2 of the demo
% of a stencil whose active region moves (docs/INPROGRESS.md). Each node owns
% half of the grid's rows, on its GPU, and after each step sends its edge row
% to the other and takes the other's into its halo, from a target region on
% the GPU. The cells must be life.m's. See README.md.
%
% Node 2 is `life_two partner`, which listens on loopback and prints its port;
% node 1 is `life_two first PORT`, which connects to it, and prints the whole
% grid's population and checksum and the step times. pair.py runs both.

:- module life_two.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, maybe, process, string.

    % A row of the grid, in a message: W cells, then the step it is from,
    % which the receiver checks.
    %
:- type row == array(int).

main(!IO) :-
    setting("LIFE_SIZE", 4096, N, !IO),
    setting("LIFE_STEPS", 100, Steps, !IO),
    io.command_line_arguments(Arguments, !IO),
    ( if N < 2 then
        usage(!IO)
    else if Steps < 2 then
        usage(!IO)
    else if Arguments = ["partner"] then
        partner(N, Steps, !IO)
    else if Arguments = ["first", Port] then
        first(N, Steps, Port, !IO)
    else
        usage(!IO)
    ).

:- pred usage(io::di, io::uo) is det.
usage(!IO) :-
    io.write_string("usage: life_two partner | life_two first PORT\n" ++
        "LIFE_SIZE must be at least 2 and LIFE_STEPS at least 2.\n", !IO),
    exit_failure(!IO).

    % Node 2: the bottom rows. It waits for node 1's mailboxes, for its rows
    % and for the results, and sends its rows to `halo`.
    %
:- pred partner(int::in, int::in, io::di, io::uo) is det.
partner(N, Steps, !IO) :-
    new_mailbox(Halo : mailbox(row), !IO),
    new_mailbox(Hello : mailbox(mailbox(row)), !IO),
    register("halo", Halo, _, !IO),
    register("hello", Hello, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        receive(Hello, any, after(60000), GotOut, !IO),
        receive(Hello, any, after(60000), GotResults, !IO),
        ( if
            GotOut = message(Out),
            GotResults = message(Results)
        then
            Top = N / 2,
            simulate(N, Steps, Top, N - Top, Out, Halo, Ok, Population,
                Checksum, _, _, !IO),
            send(Results, array.from_list([Ok, Population, Checksum]), !IO)
        else
            io.write_string("node 1 did not say hello\n", !IO),
            exit_failure(!IO)
        )
    ;
        Listening = error(Error),
        io.write_string("listen: " ++ Error ++ "\n", !IO),
        exit_failure(!IO)
    ).

    % Node 1: the top rows. It adds node 2's population and checksum to its
    % own, and prints them with its step times.
    %
:- pred first(int::in, int::in, string::in, io::di, io::uo) is det.
first(N, Steps, Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Node),
        lookup(Node, "halo", after(10000), FoundOut, !IO),
        lookup(Node, "hello", after(10000), FoundHello, !IO),
        ( if
            FoundOut = ok(Out),
            FoundHello = ok(Hello)
        then
            new_mailbox(Halo : mailbox(row), !IO),
            new_mailbox(Results : mailbox(row), !IO),
            send(Hello, Halo, !IO),
            send(Hello, Results, !IO),
            header(N, Steps, !IO),
            simulate(N, Steps, 0, N / 2, Out, Halo, Ok, Population,
                Checksum, Times, Exchanges, !IO),
            receive(Results, any, after(60000), Got, !IO),
            ( if
                Ok = 1,
                Got = message(Theirs),
                Theirs ^ elem(0) = 1
            then
                summary(N, Steps, Times, Exchanges, Population + Theirs ^ elem(1),
                    Checksum + Theirs ^ elem(2), !IO)
            else
                io.write_string("the halo exchange failed\n", !IO),
                exit_failure(!IO)
            )
        else
            io.write_string("lookups failed\n", !IO),
            exit_failure(!IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO),
        exit_failure(!IO)
    ).

    % A node's part: H rows of the grid from row R0, in an array of H + 2
    % rows of N + 2 cells, the first and last its halo, or the grid's border
    % where it has no neighbour there. Out is the other node's mailbox for
    % rows, In this node's. Ok is 1 if every exchange was right; Times is
    % each step's time, the exchange included, and Exchanges each exchange's.
    %
:- pred simulate(int::in, int::in, int::in, int::in, mailbox(row)::in,
    mailbox(row)::in, int::out, int::out, int::out, array(float)::out,
    array(float)::out, io::di, io::uo) is det.
simulate(N, Steps, R0, H, Out, In, Ok, Population, Checksum, Times,
        Exchanges, !IO) :-
    W = N + 2,
    init(N, R0, H, array.init((H + 2) * W, 0), A),
    array.init((H + 2) * W, 0, B),
    array.init(Steps, 0.0, T0),
    array.init(Steps, 0.0, X0),
    %$omp target data map(tofrom: A, B)
    (
        exchange(N, R0, 0, Out, In, Ok0, A, A1, !IO),
        run(0, Steps, N, R0, H, Out, In, Ok0, Ok, A1, B, Final, T0, Times,
            X0, Exchanges, !IO)
    ),
    count(N, R0, H, Final, Population, Checksum).

:- pred edge_rows(int::in, int::in, int::out, int::out) is det.
edge_rows(N, R0, Send, Receive) :-
    % Node 1, from row 0, sends its last row and receives below it; node 2
    % sends its first row and receives above it.
    ( if R0 = 0 then
        Send = N / 2,
        Receive = N / 2 + 1
    else
        Send = 1,
        Receive = 0
    ).

    % Whether the cell of index K, counting the grid's cells by rows from 0,
    % starts alive, as in life.m.
    %
:- func alive(int) = int.
alive(K) = V :-
    Hash = (K * 2654435761) /\ 4294967295,
    ( if unchecked_rem(Hash >> 16, 100) < 30 then V = 1 else V = 0 ).

:- pred init(int::in, int::in, int::in,
    array(int)::array_di, array(int)::array_uo) is det.
init(N, R0, H, !A) :-
    W = N + 2,
    %$omp target teams distribute parallel do
    ( for(K, 0, H * N - 1) do
        C = (unchecked_quotient(K, N) + 1) * W + unchecked_rem(K, N) + 1,
        !A ^ unsafe_elem(C) := alive(R0 * N + K)
    ).

    % One step of the node's rows: B's cells from A's, as in life.m.
    %
:- pred step(int::in, int::in, array(int)::array_ui,
    array(int)::array_di, array(int)::array_uo) is det.
step(N, H, A, !B) :-
    W = N + 2,
    %$omp target teams distribute parallel do
    ( for(K, 0, H * N - 1) do
        C = (unchecked_quotient(K, N) + 1) * W + unchecked_rem(K, N) + 1,
        Count =
            A ^ unsafe_elem(C - W - 1) + A ^ unsafe_elem(C - W) +
            A ^ unsafe_elem(C - W + 1) + A ^ unsafe_elem(C - 1) +
            A ^ unsafe_elem(C + 1) + A ^ unsafe_elem(C + W - 1) +
            A ^ unsafe_elem(C + W) + A ^ unsafe_elem(C + W + 1),
        ( if Count = 3 then
            V = 1
        else if Count = 2 then
            V = A ^ unsafe_elem(C)
        else
            V = 0
        ),
        !B ^ unsafe_elem(C) := V
    ).

    % The exchange after step S, on the GPU: the node's edge row to the
    % other node, and the other's into the halo. Ok is 1 if the row came,
    % from step S.
    %
:- pred exchange(int::in, int::in, int::in, mailbox(row)::in,
    mailbox(row)::in, int::out,
    array(int)::array_di, array(int)::array_uo, io::di, io::uo) is det.
exchange(N, R0, S, Out, In, Ok, !A, !IO) :-
    W = N + 2,
    edge_rows(N, R0, SendRow, ReceiveRow),
    %$omp target
    (
        edge(W, SendRow * W, S, !.A, Edge),
        send(Out, Edge, !IO),
        receive(In, any, after(60000), Got, !IO),
        ( if
            Got = message(Row),
            array.size(Row) = W + 1,
            Row ^ elem(W) = S
        then
            fill(W, ReceiveRow * W, Row, !A),
            Ok = 1
        else
            Ok = 0
        )
    ).

    % The W cells from From, and S after them.
    %
:- pred edge(int::in, int::in, int::in, array(int)::array_ui, row::array_uo)
    is det.
edge(W, From, S, A, Edge) :-
    array.init(W + 1, S, Edge0),
    ( for(I, 0, W - 1), fromto(Edge0, E0, E1, Edge) do
        E1 = E0 ^ unsafe_elem(I) := A ^ unsafe_elem(From + I)
    ).

:- pred fill(int::in, int::in, row::in,
    array(int)::array_di, array(int)::array_uo) is det.
fill(W, To, Row, !A) :-
    ( for(I, 0, W - 1) do
        !A ^ unsafe_elem(To + I) := Row ^ unsafe_elem(I)
    ).

    % Steps from S to Steps, as in life.m, each followed by the exchange.
    %
:- pred run(int::in, int::in, int::in, int::in, int::in, mailbox(row)::in,
    mailbox(row)::in, int::in, int::out,
    array(int)::array_di, array(int)::array_di, array(int)::array_uo,
    array(float)::array_di, array(float)::array_uo,
    array(float)::array_di, array(float)::array_uo, io::di, io::uo) is det.
run(S, Steps, N, R0, H, Out, In, !Ok, Cur, Next0, Final, !T, !X, !IO) :-
    ( if S >= Steps then
        Final = Cur
    else if !.Ok = 0 then
        Final = Cur
    else
        now(T0, !IO),
        step(N, H, Cur, Next0, Next1),
        now(T1, !IO),
        exchange(N, R0, S + 1, Out, In, !:Ok, Next1, Next, !IO),
        now(T2, !IO),
        !T ^ elem(S) := T2 - T0,
        !X ^ elem(S) := T2 - T1,
        run(S + 1, Steps, N, R0, H, Out, In, !Ok, Next, Cur, Final, !T, !X,
            !IO)
    ).

    % The node's live cells, and the sum of their indices in the whole grid.
    %
:- pred count(int::in, int::in, int::in, array(int)::in, int::out, int::out)
    is det.
count(N, R0, H, A, Population, Checksum) :-
    W = N + 2,
    ( for(K, 0, H * N - 1),
        fromto(0, P0, P1, Population), fromto(0, S0, S1, Checksum) do
        C = (K / N + 1) * W + K mod N + 1,
        V = A ^ elem(C),
        P1 = P0 + V,
        S1 = S0 + V * (R0 * N + K)
    ).

:- pred summary(int::in, int::in, array(float)::in, array(float)::in,
    int::in, int::in, io::di, io::uo) is det.
summary(N, Steps, T, X, Population, Checksum, !IO) :-
    ( for(S, 1, Steps - 1),
        fromto(T ^ elem(1), Min0, Min1, Min),
        fromto(T ^ elem(1), Max0, Max1, Max),
        fromto(0.0, Sum0, Sum1, Sum),
        fromto(T ^ elem(1), XMin0, XMin1, XMin),
        fromto(0.0, XSum0, XSum1, XSum) do
        Min1 = float.min(Min0, T ^ elem(S)),
        Max1 = float.max(Max0, T ^ elem(S)),
        Sum1 = Sum0 + T ^ elem(S),
        XMin1 = float.min(XMin0, X ^ elem(S)),
        XSum1 = XSum0 + X ^ elem(S)
    ),
    Average = Sum / float(Steps - 1),
    Rate = float(N) * float(N) / Min * 1.0e-9,
    write_results(Population, Checksum, Rate, Min, Max, Average, !IO),
    write_exchange(XMin, XSum / float(Steps - 1), !IO).

:- pragma foreign_decl("C", "#include <stdio.h>").
:- pragma foreign_decl("C", "#include <stdlib.h>").
:- pragma foreign_decl("C", "#include <time.h>").

:- pred setting(string::in, int::in, int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    setting(Name::in, Default::in, Value::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    const char *s = getenv(Name);
    Value = s && *s ? (MR_Integer) strtol(s, NULL, 10) : Default;
").

:- pred exit_failure(io::di, io::uo) is det.
:- pragma foreign_proc("C",
    exit_failure(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    fflush(stdout);
    exit(1);
").

:- pred now(float::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    now(T::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    T = (double) ts.tv_sec + 1.0e-9 * (double) ts.tv_nsec;
").

:- pred header(int::in, int::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    header(N::in, Steps::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Game of Life, Mercury, two nodes\\n\");
    printf(\"Grid: %ld x %ld cells, %ld bytes a cell\\n\", (long) N, (long) N,
        (long) sizeof(MR_Integer));
    printf(\"Steps: %ld\\n\", (long) Steps);
").

:- pred write_results(int::in, int::in, float::in, float::in, float::in,
    float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_results(Population::in, Checksum::in, Rate::in, Min::in, Max::in,
        Average::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Population: %ld\\n\", (long) Population);
    printf(\"Checksum: %ld\\n\", (long) Checksum);
    printf(\"%-14s%-12s%-12s%-12s\\n\", \"Gcells/sec\", \"Min (sec)\", \"Max\",
        \"Average\");
    printf(\"%-14.3f%-12.6f%-12.6f%-12.6f\\n\", Rate, Min, Max, Average);
").

    % The exchanges' part of the steps, as their min and average.
    %
:- pred write_exchange(float::in, float::in, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    write_exchange(Min::in, Average::in, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    printf(\"Exchange: min %.6f, average %.6f sec\\n\", Min, Average);
").
