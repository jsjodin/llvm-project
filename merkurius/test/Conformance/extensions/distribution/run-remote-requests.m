% DESIGN: Distribution and wire format / Remote calls
% Processes on two nodes exchange messages with the same send and receive
% as on one. The server registers its mailbox, and the client connects,
% looks the name up, and finds a mailbox of the server's node. Requests
% carry the mailbox to reply to: values of many kinds come back unchanged,
% as a remote send encodes them and the receiver decodes them by its own
% types; pipelined requests are answered in order, as messages between two
% processes arrive in order; and a reply can go to a third mailbox. The
% oracle is a translation into distributed Erlang, which the design follows.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/remote_requests_translation.erl
% NODES: server client
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %nodes %t2 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %nodes %t2 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:server: served 1009 requests
% OUTPUT-NEXT:client: connected
% OUTPUT-NEXT:client: the server is another node: yes
% OUTPUT-NEXT:client: echo is on the server: yes
% OUTPUT-NEXT:client: echoed: message(int(42))
% OUTPUT-NEXT:client: echoed: message(int(-9223372036854775808))
% OUTPUT-NEXT:client: echoed: message(text(""))
% OUTPUT-NEXT:client: echoed: message(text("a string with \"quotes\" and a\nnewline"))
% OUTPUT-NEXT:client: echoed: message(floats([0.5, -1.25, 100.0]))
% OUTPUT-NEXT:client: echoed: message(pair(int(1), pair(text("two"), none)))
% OUTPUT-NEXT:client: echoed: message(values([none, int(3), values([]), values([text("x")])]))
% OUTPUT-NEXT:client: echoed: message(values([int(1), int(4), int(9), int(16), int(25), int(36), int(49), int(64), int(81), int(100), int(121), int(144)]))
% OUTPUT-NEXT:client: 1000 pipelined replies in order: yes
% OUTPUT-NEXT:client: message(text("to a third mailbox"))
% OUTPUT-NOT:{{.}}
:- module run_remote_requests.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, list, maybe, process, string.

:- type value
    --->    int(int)
    ;       text(string)
    ;       floats(list(float))
    ;       pair(value, value)
    ;       values(list(value))
    ;       none.

:- type request
    --->    echo(value, mailbox(value))
    ;       sum(int, list(int), mailbox(value))
    ;       stop.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["server"] then
        server(!IO)
    else if Args = ["client", Address] then
        client(Address, !IO)
    else
        io.write_string("usage: server | client ADDRESS\n", !IO)
    ).

:- pred server(io::di, io::uo) is det.

server(!IO) :-
    listen("127.0.0.1:0", Listened, !IO),
    (
        Listened = ok(Port),
        io.format("listening 127.0.0.1:%d\n", [i(Port)], !IO),
        io.flush_output(!IO),
        new_mailbox(Requests, !IO),
        register("echo", Requests, Registered, !IO),
        (
            Registered = ok,
            serve(Requests, 0, !IO)
        ;
            Registered = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    ;
        Listened = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred serve(mailbox(request)::in, int::in, io::di, io::uo) is det.

serve(Requests, Served, !IO) :-
    receive(Requests, any, infinity, R, !IO),
    (
        R = message(echo(Value, Reply)),
        send(Reply, Value, !IO),
        serve(Requests, Served + 1, !IO)
    ;
        R = message(sum(N, Ints, Reply)),
        send(Reply, pair(int(N), int(list.foldl(plus, Ints, 0))), !IO),
        serve(Requests, Served + 1, !IO)
    ;
        R = message(stop),
        io.format("served %d requests\n", [i(Served)], !IO)
    ;
        R = timeout
    ).

:- pred client(string::in, io::di, io::uo) is det.

client(Address, !IO) :-
    connect(Address, Connected, !IO),
    (
        Connected = ok(Server),
        io.write_string("connected\n", !IO),
        this_node(Here, !IO),
        io.format("the server is another node: %s\n",
            [s(if Server = Here then "no" else "yes")], !IO),
        lookup(Server, "echo", after(5000), Found, !IO),
        (
            Found = ok(Echo),
            io.format("echo is on the server: %s\n",
                [s(if node_of(Echo) = Server then "yes" else "no")], !IO),
            requests(Echo, !IO)
        ;
            Found = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    ;
        Connected = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred requests(mailbox(request)::in, io::di, io::uo) is det.

requests(Echo, !IO) :-
    new_mailbox(Replies, !IO),
    Values = [
        int(42),
        int(-9223372036854775808),
        text(""),
        text("a string with \"quotes\" and a\nnewline"),
        floats([0.5, -1.25, 100.0]),
        pair(int(1), pair(text("two"), none)),
        values([none, int(3), values([]), values([text("x")])]),
        values(list.map(func(I) = int(I * I), 1 .. 12))
    ],
    list.foldl(echo(Echo, Replies), Values, !IO),
    list.foldl(send_sum(Echo, Replies), 1 .. 1000, !IO),
    replies_in_order(Replies, 1, 1000, yes, InOrder, !IO),
    io.format("1000 pipelined replies in order: %s\n", [s(yes_no(InOrder))],
        !IO),
    new_mailbox(Other, !IO),
    send(Echo, echo(text("to a third mailbox"), Other), !IO),
    receive(Other, any, after(10000), R, !IO),
    io.write_line(R, !IO),
    send(Echo, stop, !IO).

:- pred echo(mailbox(request)::in, mailbox(value)::in, value::in,
    io::di, io::uo) is det.

echo(Echo, Replies, Value, !IO) :-
    send(Echo, echo(Value, Replies), !IO),
    receive(Replies, any, after(10000), R, !IO),
    ( if R = message(Value) then
        io.write_string("echoed: ", !IO)
    else
        io.write_string("changed: ", !IO)
    ),
    io.write_line(R, !IO).

:- pred send_sum(mailbox(request)::in, mailbox(value)::in, int::in,
    io::di, io::uo) is det.

send_sum(Echo, Replies, N, !IO) :-
    send(Echo, sum(N, 1 .. N, Replies), !IO).

:- pred replies_in_order(mailbox(value)::in, int::in, int::in, bool::in,
    bool::out, io::di, io::uo) is det.

replies_in_order(Replies, N, Last, !InOrder, !IO) :-
    ( if N > Last then
        true
    else
        receive(Replies, any, after(10000), R, !IO),
        ( if R = message(pair(int(N), int(N * (N + 1) // 2))) then
            true
        else
            !:InOrder = no
        ),
        replies_in_order(Replies, N + 1, Last, !InOrder, !IO)
    ).

:- func yes_no(bool) = string.

yes_no(yes) = "yes".
yes_no(no) = "no".
