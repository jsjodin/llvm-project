% Two nodes over loopback (docs/INPROGRESS.md, Remote handles and TCP), for
% which mmc cannot be the oracle, as it has no process module. A driver
% runs the program in its roles, one process per node; see
% Inputs/two-nodes.py.
%
% - connect's and listen's errors: no address, nobody listening, a port in
%   use, and a node connecting to itself.
% - lookup on another node: an unknown name, a mismatched type, and a
%   found mailbox, which is on that node.
% - Requests with reply handles: round trips, pipelined requests whose
%   replies come in order, large lists and arrays, and terms of many kinds
%   echoed unchanged; a reply to a third mailbox.
% - Ticks keep an idle connection; a stopped peer is lost within a few tick
%   times, which fails a waiting lookup.
% - A closed mailbox drops later messages, and its name goes; once the
%   other node quits, lookups fail and sends are dropped.
% - Two nodes connecting to each other at once end with one connection,
%   and messages sent while that is decided arrive in order.
% - The cookie: a node with another one is refused; ~/.merkurius.cookie is
%   made when it is missing, and then used; the cookie is its first line;
%   a file that others can read, an empty one, an empty MERK_COOKIE and no
%   HOME are errors.
%
% RUN: merkc -O0 %s -o %t0 && %python %S/Inputs/two-nodes.py %t0 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %python %S/Inputs/two-nodes.py %t2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% OUTPUT:pingpong:
% OUTPUT-NEXT:client: connected
% OUTPUT-NEXT:client: the server is another node: yes
% OUTPUT-NEXT:client: connect to nonsense: not an address of the form host:port: nonsense
% OUTPUT-NEXT:client: connect to port 1: 127.0.0.1:1: Connection refused
% OUTPUT-NEXT:client: listen on the server's port: 127.0.0.1:PORT: Address already in use
% OUTPUT-NEXT:client: connect to itself: 127.0.0.1:PORT: a node cannot connect to itself
% OUTPUT-NEXT:client: nobody: no mailbox is registered as "nobody"
% OUTPUT-NEXT:client: pong as mailbox(string): the mailbox registered as "pong" has messages of another type
% OUTPUT-NEXT:client: pong as mailbox(request): found
% OUTPUT-NEXT:client: pong is on the server: yes
% OUTPUT-NEXT:client: round trips answered: 1000
% OUTPUT-NEXT:client: 10000 pipelined replies in order: yes
% OUTPUT-NEXT:client: sums: yes(sums(5000050000, 2499975000.0))
% OUTPUT-NEXT:client: echoes equal: 300 of 300
% OUTPUT-NEXT:client: reply to another mailbox: yes(pong(7))
% OUTPUT-NEXT:client: and to the first: no
% OUTPUT-NEXT:client: idle for three seconds: no
% OUTPUT-NEXT:client: then: yes(pong(8))
% OUTPUT-NEXT:client: stop: yes(stopping)
% OUTPUT-NEXT:client: the closed mailbox answers: no
% OUTPUT-NEXT:client: pong after its close: no mailbox is registered as "pong"
% OUTPUT-NEXT:client: quit: yes(stopping)
% OUTPUT-NEXT:client: the server is gone: yes
% OUTPUT-NEXT:client: after it is gone: no
% OUTPUT-NEXT:server: listening
% OUTPUT-NEXT:server: register pong: ok
% OUTPUT-NEXT:server: pong closed
% OUTPUT-NEXT:server: quitting
% OUTPUT-NEXT:stall:
% OUTPUT-NEXT:client: connected
% OUTPUT-NEXT:client: lookup after the stop: the connection to the node was lost
% OUTPUT-NEXT:lost within 0.5 to 10 seconds: yes
% OUTPUT-NEXT:simultaneous:
% OUTPUT-NEXT:20 rounds as expected
% OUTPUT-NEXT:cookie:
% OUTPUT-NEXT:another cookie: connect: 127.0.0.1:PORT: the other node has another cookie
% OUTPUT-NEXT:the same cookie: connect: ok
% OUTPUT-NEXT:no file, listen: listen: ok
% OUTPUT-NEXT:made: mode 400, 20 capital letters: yes
% OUTPUT-NEXT:the file's cookie: connect: ok
% OUTPUT-NEXT:it in MERK_COOKIE: connect: ok
% OUTPUT-NEXT:first line: connect: ok
% OUTPUT-NEXT:both lines: connect: 127.0.0.1:PORT: the other node has another cookie
% OUTPUT-NEXT:readable by its group: listen: no cookie: HOME/.merkurius.cookie: the cookie file must be its owner's and readable by its owner only
% OUTPUT-NEXT:empty: listen: no cookie: HOME/.merkurius.cookie: the cookie file is empty
% OUTPUT-NEXT:empty MERK_COOKIE: listen: no cookie: MERK_COOKIE: the cookie is empty
% OUTPUT-NEXT:no HOME: listen: no cookie: ~/.merkurius.cookie: HOME is not set, and neither is MERK_COOKIE
% OUTPUT-NOT:{{.}}

:- module process_remote.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, bool, char, float, int, int8, int32, list, maybe.
:- import_module process, string, uint16, uint64.

:- type request
    --->    ping(int, mailbox(reply))
    ;       sum(list(int), array(float), mailbox(reply))
    ;       echo(data, mailbox(reply))
    ;       stop(mailbox(reply)).

:- type reply
    --->    pong(int)
    ;       sums(int, float)
    ;       echoed(data)
    ;       stopping.

:- type data
    --->    leaf
    ;       text(string, char)
    ;       numbers(int8, uint16, int32, uint64, float)
    ;       pair(data, data)
    ;       many(list(data))
    ;       optional(maybe(int))
    ;       tuple({int, string}).

:- type control
    --->    quit(mailbox(reply)).

:- type peer_message
    --->    peer_ping(int, mailbox(peer_message))
    ;       peer_pong(int)
    ;       peer_done
    ;       peer_bye.

:- pragma foreign_decl("C", local, "
extern MR_Integer merk_node_connection_count(void);
").

:- pred connection_count(int::out, io::di, io::uo) is det.
:- pragma foreign_proc("C",
    connection_count(Count::out, _IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure, thread_safe],
"
    Count = merk_node_connection_count();
").

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["server"] then
        server(!IO)
    else if Arguments = ["client", Port] then
        client(Port, !IO)
    else if Arguments = ["stall", Port] then
        stall(Port, !IO)
    else if Arguments = ["peer"] then
        peer(!IO)
    else if Arguments = ["probe", Port] then
        probe(Port, !IO)
    else if Arguments = ["listen"] then
        listen_only(!IO)
    else
        io.write_string("usage: server | client PORT | stall PORT | peer | " ++
            "probe PORT | listen\n", !IO),
        io.set_exit_status(2, !IO)
    ).

%---------------------------------------------------------------------------%
% The server.

:- pred server(io::di, io::uo) is det.

server(!IO) :-
    new_mailbox(Pong : mailbox(request), !IO),
    new_mailbox(Control : mailbox(control), !IO),
    register("pong", Pong, Registered, !IO),
    register("control", Control, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        io.write_string("register pong: ", !IO),
        io.write(Registered, !IO),
        io.nl(!IO),
        serve(Pong, !IO),
        close_mailbox(Pong, !IO),
        io.write_string("pong closed\n", !IO),
        receive(Control, any, infinity, Quit, !IO),
        (
            Quit = yes(quit(Reply)),
            send(Reply, stopping, !IO),
            io.write_string("quitting\n", !IO)
        ;
            Quit = no,
            io.write_string("no quit\n", !IO)
        )
    ;
        Listening = error(Error1),
        io.write_string("listen: " ++ Error1 ++ "\n", !IO)
    ).

:- pred serve(mailbox(request)::in, io::di, io::uo) is det.

serve(Pong, !IO) :-
    receive(Pong, any, infinity, Got, !IO),
    (
        Got = yes(Request),
        (
            Request = ping(N, Reply),
            send(Reply, pong(N), !IO),
            serve(Pong, !IO)
        ;
            Request = sum(Ints, Floats, Reply),
            send(Reply, sums(list.foldl(add_int, Ints, 0),
                array.foldl(add_float, Floats, 0.0)), !IO),
            serve(Pong, !IO)
        ;
            Request = echo(Data, Reply),
            send(Reply, echoed(Data), !IO),
            serve(Pong, !IO)
        ;
            Request = stop(Reply),
            send(Reply, stopping, !IO)
        )
    ;
        Got = no
    ).

:- func add_int(int, int) = int.

add_int(X, Y) = X + Y.

:- func add_float(float, float) = float.

add_float(X, Y) = X + Y.

%---------------------------------------------------------------------------%
% The client.

:- pred client(string::in, io::di, io::uo) is det.

client(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Server),
        io.write_string("connected\n", !IO),
        this_node(Self, !IO),
        yes_no("the server is another node", ( if Server = Self then no else yes ),
            !IO),
        address_errors(Port, !IO),
        lookup(Server, "nobody", after(10000),
            Nobody : maybe_error(mailbox(request)), !IO),
        show_lookup("nobody", Nobody, !IO),
        lookup(Server, "pong", after(10000),
            AsString : maybe_error(mailbox(string)), !IO),
        show_lookup("pong as mailbox(string)", AsString, !IO),
        lookup(Server, "pong", after(10000), Found, !IO),
        show_lookup("pong as mailbox(request)", Found, !IO),
        (
            Found = ok(Pong),
            yes_no("pong is on the server",
                ( if node_of(Pong) = Server then yes else no ), !IO),
            talk(Server, Pong, !IO)
        ;
            Found = error(_)
        )
    ;
        Connected = error(Error1),
        io.write_string("connect: " ++ Error1 ++ "\n", !IO)
    ).

:- pred address_errors(string::in, io::di, io::uo) is det.

address_errors(ServerPort, !IO) :-
    connect("nonsense", NoAddress, !IO),
    show_connect("connect to nonsense", NoAddress, !IO),
    connect("127.0.0.1:1", NobodyThere, !IO),
    show_connect("connect to port 1", NobodyThere, !IO),
    listen("127.0.0.1:" ++ ServerPort, InUse, !IO),
    (
        InUse = ok(_),
        io.write_string("listen on the server's port: ok\n", !IO)
    ;
        InUse = error(Error1),
        io.write_string("listen on the server's port: " ++
            replace_all(Error1, ServerPort, "PORT") ++ "\n", !IO)
    ),
    listen("127.0.0.1:0", Own, !IO),
    (
        Own = ok(OwnPort),
        connect("127.0.0.1:" ++ int_to_string(OwnPort), Itself, !IO),
        (
            Itself = ok(_),
            io.write_string("connect to itself: ok\n", !IO)
        ;
            Itself = error(Error2),
            io.write_string("connect to itself: " ++
                replace_all(Error2, int_to_string(OwnPort), "PORT") ++ "\n",
                !IO)
        )
    ;
        Own = error(Error3),
        io.write_string("listen: " ++ Error3 ++ "\n", !IO)
    ).

:- pred talk(node::in, mailbox(request)::in, io::di, io::uo) is det.

talk(Server, Pong, !IO) :-
    new_mailbox(Replies : mailbox(reply), !IO),
    round_trips(Pong, Replies, 1, 1000, 0, Good, !IO),
    io.write_string("round trips answered: " ++ int_to_string(Good) ++ "\n",
        !IO),
    send_pings(Pong, Replies, 1, 10000, !IO),
    receive_pongs(Replies, 1, 10000, yes, InOrder, !IO),
    yes_no("10000 pipelined replies in order", InOrder, !IO),
    Floats = array.generate(100000, half),
    send(Pong, sum(1 .. 100000, Floats, Replies), !IO),
    receive(Replies, any, after(10000), Sums, !IO),
    io.write_string("sums: ", !IO),
    io.write(Sums, !IO),
    io.nl(!IO),
    make_data_list(300, 0, Datas, 12345, _),
    send_echoes(Pong, Replies, Datas, !IO),
    receive_echoes(Replies, Datas, 0, Equal, !IO),
    io.write_string("echoes equal: " ++ int_to_string(Equal) ++ " of " ++
        int_to_string(list.length(Datas)) ++ "\n", !IO),
    new_mailbox(Other : mailbox(reply), !IO),
    send(Pong, ping(7, Other), !IO),
    receive(Other, any, after(10000), ToOther, !IO),
    io.write_string("reply to another mailbox: ", !IO),
    io.write(ToOther, !IO),
    io.nl(!IO),
    receive(Replies, any, after(0), ToFirst, !IO),
    io.write_string("and to the first: ", !IO),
    io.write(ToFirst, !IO),
    io.nl(!IO),
    receive(Replies, any, after(3000), Idle, !IO),
    io.write_string("idle for three seconds: ", !IO),
    io.write(Idle, !IO),
    io.nl(!IO),
    send(Pong, ping(8, Replies), !IO),
    receive(Replies, any, after(10000), AfterIdle, !IO),
    io.write_string("then: ", !IO),
    io.write(AfterIdle, !IO),
    io.nl(!IO),
    send(Pong, stop(Replies), !IO),
    receive(Replies, any, after(10000), Stopped, !IO),
    io.write_string("stop: ", !IO),
    io.write(Stopped, !IO),
    io.nl(!IO),
    send(Pong, ping(9, Replies), !IO),
    receive(Replies, any, after(500), FromClosed, !IO),
    io.write_string("the closed mailbox answers: ", !IO),
    io.write(FromClosed, !IO),
    io.nl(!IO),
    lookup(Server, "pong", after(10000),
        Closed : maybe_error(mailbox(request)), !IO),
    show_lookup("pong after its close", Closed, !IO),
    lookup(Server, "control", after(10000), Control, !IO),
    (
        Control = ok(ControlBox),
        send(ControlBox, quit(Replies), !IO),
        receive(Replies, any, after(10000), Quit, !IO),
        io.write_string("quit: ", !IO),
        io.write(Quit, !IO),
        io.nl(!IO),
        wait_gone(Server, 100, Gone, !IO),
        yes_no("the server is gone", Gone, !IO),
        send(Pong, ping(10, Replies), !IO),
        send(ControlBox, quit(Replies), !IO),
        receive(Replies, any, after(200), AfterGone, !IO),
        io.write_string("after it is gone: ", !IO),
        io.write(AfterGone, !IO),
        io.nl(!IO)
    ;
        Control = error(Error1),
        io.write_string("control: " ++ Error1 ++ "\n", !IO)
    ).

:- func half(int) = float.

half(I) = float(I) * 0.5.

:- pred round_trips(mailbox(request)::in, mailbox(reply)::in, int::in,
    int::in, int::in, int::out, io::di, io::uo) is det.

round_trips(Pong, Replies, I, N, !Good, !IO) :-
    ( if I > N then
        true
    else
        send(Pong, ping(I, Replies), !IO),
        receive(Replies, any, after(10000), Reply, !IO),
        ( if Reply = yes(pong(I)) then
            !:Good = !.Good + 1
        else
            true
        ),
        round_trips(Pong, Replies, I + 1, N, !Good, !IO)
    ).

:- pred send_pings(mailbox(request)::in, mailbox(reply)::in, int::in,
    int::in, io::di, io::uo) is det.

send_pings(Pong, Replies, I, N, !IO) :-
    ( if I > N then
        true
    else
        send(Pong, ping(I, Replies), !IO),
        send_pings(Pong, Replies, I + 1, N, !IO)
    ).

:- pred receive_pongs(mailbox(reply)::in, int::in, int::in, bool::in,
    bool::out, io::di, io::uo) is det.

receive_pongs(Replies, I, N, !InOrder, !IO) :-
    ( if I > N then
        true
    else
        receive(Replies, any, after(10000), Reply, !IO),
        ( if Reply = yes(pong(I)) then
            true
        else
            !:InOrder = no
        ),
        receive_pongs(Replies, I + 1, N, !InOrder, !IO)
    ).

:- pred send_echoes(mailbox(request)::in, mailbox(reply)::in,
    list(data)::in, io::di, io::uo) is det.

send_echoes(_, _, [], !IO).
send_echoes(Pong, Replies, [Data | Datas], !IO) :-
    send(Pong, echo(Data, Replies), !IO),
    send_echoes(Pong, Replies, Datas, !IO).

:- pred receive_echoes(mailbox(reply)::in, list(data)::in, int::in,
    int::out, io::di, io::uo) is det.

receive_echoes(_, [], !Equal, !IO).
receive_echoes(Replies, [Data | Datas], !Equal, !IO) :-
    receive(Replies, any, after(10000), Reply, !IO),
    ( if Reply = yes(echoed(Data)) then
        !:Equal = !.Equal + 1
    else
        true
    ),
    receive_echoes(Replies, Datas, !Equal, !IO).

    % Waits until a lookup finds that the connection is gone.
:- pred wait_gone(node::in, int::in, bool::out, io::di, io::uo) is det.

wait_gone(Server, Tries, Gone, !IO) :-
    lookup(Server, "control", after(1000),
        Result : maybe_error(mailbox(control)), !IO),
    ( if
        Result = error(Error1),
        ( Error1 = "not connected to the node"
        ; Error1 = "the connection to the node was lost"
        )
    then
        Gone = yes
    else if Tries =< 0 then
        Gone = no
    else
        pause(100, !IO),
        wait_gone(Server, Tries - 1, Gone, !IO)
    ).

%---------------------------------------------------------------------------%
% Terms of many kinds, from a linear congruential generator.

:- pred next(int::out, int::in, int::out) is det.

next(N, Seed0, Seed) :-
    Seed = (Seed0 * 1103515245 + 12345) mod 2147483648,
    N = Seed.

:- pred make_data_list(int::in, int::in, list(data)::out, int::in, int::out)
    is det.

make_data_list(Count, Depth, Datas, !Seed) :-
    ( if Count =< 0 then
        Datas = []
    else
        make_data(Depth, Data, !Seed),
        make_data_list(Count - 1, Depth, Rest, !Seed),
        Datas = [Data | Rest]
    ).

:- pred make_data(int::in, data::out, int::in, int::out) is det.

make_data(Depth, Data, !Seed) :-
    next(N, !Seed),
    Kind = ( if Depth >= 4 then N mod 4 else N mod 7 ),
    ( if Kind = 0 then
        Data = leaf
    else if Kind = 1 then
        next(A, !Seed),
        Data = text(int_to_string(A) ++ " é€😀",
            char.det_from_int(0x3B1 + A mod 20))
    else if Kind = 2 then
        next(A, !Seed),
        next(B, !Seed),
        Data = numbers(int8.cast_from_int(A), uint16.cast_from_int(B),
            int32.cast_from_int(A - 1073741824), uint64.cast_from_int(A * B),
            float(A) / 7.0)
    else if Kind = 3 then
        Data = optional(( if N mod 2 = 0 then no else yes(N - 1000000000) ))
    else if Kind = 4 then
        make_data(Depth + 1, X, !Seed),
        make_data(Depth + 1, Y, !Seed),
        Data = pair(X, Y)
    else if Kind = 5 then
        next(Count, !Seed),
        make_data_list(Count mod 5, Depth + 1, Datas, !Seed),
        Data = many(Datas)
    else
        Data = tuple({N, int_to_string(N)})
    ).

%---------------------------------------------------------------------------%
% A client whose server the driver stops.

:- pred stall(string::in, io::di, io::uo) is det.

stall(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Server),
        io.write_string("connected\n", !IO),
        io.flush_output(!IO),
        io.read_line_as_string(_, !IO),
        lookup(Server, "pong", infinity,
            Found : maybe_error(mailbox(request)), !IO),
        show_lookup("lookup after the stop", Found, !IO)
    ;
        Connected = error(Error1),
        io.write_string("connect: " ++ Error1 ++ "\n", !IO)
    ).

%---------------------------------------------------------------------------%
% The cookie's checks: a node that only connects, and one that only listens.

:- pred probe(string::in, io::di, io::uo) is det.

probe(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(_),
        io.write_string("connect: ok\n", !IO)
    ;
        Connected = error(Error1),
        io.write_string("connect: " ++ replace_all(Error1, Port, "PORT") ++
            "\n", !IO)
    ).

:- pred listen_only(io::di, io::uo) is det.

listen_only(!IO) :-
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(_),
        io.write_string("listen: ok\n", !IO)
    ;
        Listening = error(Error1),
        io.write_string("listen: " ++ Error1 ++ "\n", !IO)
    ).

%---------------------------------------------------------------------------%
% Two peers that connect to each other at once.

:- pred peer(io::di, io::uo) is det.

peer(!IO) :-
    new_mailbox(Box : mailbox(peer_message), !IO),
    register("peer", Box, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        io.read_line_as_string(Line, !IO),
        ( if Line = ok(Text) then
            connect("127.0.0.1:" ++ strip(Text), Connected, !IO),
            (
                Connected = ok(Node),
                io.write_string("connected\n", !IO),
                lookup(Node, "peer", after(10000), Found, !IO),
                (
                    Found = ok(Other),
                    exchange(Box, Other, !IO)
                ;
                    Found = error(Error1),
                    io.write_string("lookup: " ++ Error1 ++ "\n", !IO)
                )
            ;
                Connected = error(Error2),
                io.write_string("connect: " ++ Error2 ++ "\n", !IO)
            )
        else
            io.write_string("no port\n", !IO)
        )
    ;
        Listening = error(Error3),
        io.write_string("listen: " ++ Error3 ++ "\n", !IO)
    ).

:- pred exchange(mailbox(peer_message)::in, mailbox(peer_message)::in,
    io::di, io::uo) is det.

exchange(Box, Other, !IO) :-
    send_peer_pings(Other, Box, 1, 100, !IO),
    serve_peer(Box, Other, 1, no, yes, InOrder, !IO),
    ( if InOrder = yes then
        io.write_string("100 pongs in order\n", !IO)
    else
        io.write_string("pongs out of order or missing\n", !IO)
    ),
    wait_connections(50, Count, !IO),
    io.write_string("connections: " ++ int_to_string(Count) ++ "\n", !IO),
    send(Other, peer_bye, !IO),
    receive(Box, any, after(20000), Bye, !IO),
    ( if Bye = yes(peer_bye) then
        io.write_string("bye\n", !IO)
    else
        io.write_string("no bye: ", !IO),
        io.write(Bye, !IO),
        io.nl(!IO)
    ).

:- pred send_peer_pings(mailbox(peer_message)::in, mailbox(peer_message)::in,
    int::in, int::in, io::di, io::uo) is det.

send_peer_pings(Other, Box, I, N, !IO) :-
    ( if I > N then
        true
    else
        send(Other, peer_ping(I, Box), !IO),
        send_peer_pings(Other, Box, I + 1, N, !IO)
    ).

    % Answers the other's pings, and takes the replies to its own, until
    % all 100 have come and the other has had all of its own.
:- pred serve_peer(mailbox(peer_message)::in, mailbox(peer_message)::in,
    int::in, bool::in, bool::in, bool::out, io::di, io::uo) is det.

serve_peer(Box, Other, Next, OtherDone, !InOrder, !IO) :-
    ( if Next > 100, OtherDone = yes then
        true
    else
        receive(Box, any, after(20000), Got, !IO),
        (
            Got = no,
            io.write_string("nothing for 20 seconds\n", !IO),
            !:InOrder = no
        ;
            Got = yes(Message),
            (
                Message = peer_ping(I, Reply),
                send(Reply, peer_pong(I), !IO),
                serve_peer(Box, Other, Next, OtherDone, !InOrder, !IO)
            ;
                Message = peer_pong(I),
                ( if I = Next then
                    true
                else
                    !:InOrder = no
                ),
                ( if Next = 100 then
                    send(Other, peer_done, !IO)
                else
                    true
                ),
                serve_peer(Box, Other, Next + 1, OtherDone, !InOrder, !IO)
            ;
                Message = peer_done,
                serve_peer(Box, Other, Next, yes, !InOrder, !IO)
            ;
                Message = peer_bye,
                io.write_string("bye before the end\n", !IO),
                !:InOrder = no
            )
        )
    ).

:- pred wait_connections(int::in, int::out, io::di, io::uo) is det.

wait_connections(Tries, Count, !IO) :-
    connection_count(Count0, !IO),
    ( if Count0 = 1 ; Tries =< 0 then
        Count = Count0
    else
        pause(100, !IO),
        wait_connections(Tries - 1, Count, !IO)
    ).

%---------------------------------------------------------------------------%

:- pred pause(int::in, io::di, io::uo) is det.

pause(Milliseconds, !IO) :-
    new_mailbox(Box : mailbox(int), !IO),
    receive(Box, any, after(Milliseconds), _, !IO),
    close_mailbox(Box, !IO).

:- pred yes_no(string::in, bool::in, io::di, io::uo) is det.

yes_no(Label, Value, !IO) :-
    io.write_string(Label ++ ": " ++
        ( if Value = yes then "yes" else "no" ) ++ "\n", !IO).

:- pred show_lookup(string::in, maybe_error(mailbox(M))::in,
    io::di, io::uo) is det.

show_lookup(Label, Result, !IO) :-
    (
        Result = ok(_),
        io.write_string(Label ++ ": found\n", !IO)
    ;
        Result = error(Error1),
        io.write_string(Label ++ ": " ++ Error1 ++ "\n", !IO)
    ).

:- pred show_connect(string::in, maybe_error(node)::in, io::di, io::uo)
    is det.

show_connect(Label, Result, !IO) :-
    (
        Result = ok(_),
        io.write_string(Label ++ ": ok\n", !IO)
    ;
        Result = error(Error1),
        io.write_string(Label ++ ": " ++ Error1 ++ "\n", !IO)
    ).
