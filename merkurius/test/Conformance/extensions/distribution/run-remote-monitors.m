% DESIGN: Distribution and wire format / Remote mailboxes, sendability and failure
% Monitors work across nodes, with Erlang's reasons. The client watches
% processes of the server that end normally, are stopped from the client,
% had ended before the monitor, or are no longer watched once demonitored.
% Then a process of the server streams numbers to the client, which come
% in order and in turn with no message of the monitor, and the server
% quits: the monitor says noconnection, and a monitor made afterwards is
% noconnection at once. The oracle is a translation into distributed Erlang.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/remote_monitors_translation.erl
% NODES: server client
%
% RUN: merkc -O0 -fprocess-declarations %s -o %t0
% RUN: %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %nodes %t0 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fprocess-declarations %s -o %t2
% RUN: %nodes %t2 server client | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:server: quitting
% OUTPUT-NEXT:client: ends: message(down(normal))
% OUTPUT-NEXT:client: stopped: message(down(killed))
% OUTPUT-NEXT:client: ends: message(down(normal))
% OUTPUT-NEXT:client: had ended: message(down(noproc))
% OUTPUT-NEXT:client: demonitored: timeout
% OUTPUT-NEXT:client: streamed: message(number(1))
% OUTPUT-NEXT:client: streamed: message(number(2))
% OUTPUT-NEXT:client: streamed: message(number(3))
% OUTPUT-NEXT:client: streamed: message(number(4))
% OUTPUT-NEXT:client: streamed: message(number(5))
% OUTPUT-NEXT:client: the server quit: message(down(noconnection))
% OUTPUT-NEXT:client: watched after the loss: message(down(noconnection))
% OUTPUT-NOT:{{.}}
:- module run_remote_monitors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, require, string.

:- type work
    --->    finish
    ;       stream(int, mailbox(int)).

:- type server_msg
    --->    worker(mailbox(mailbox(work)))
    ;       quit.

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
        register("server", Requests, _, !IO),
        serve(Requests, !IO)
    ;
        Listened = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred serve(mailbox(server_msg)::in, io::di, io::uo) is det.

serve(Requests, !IO) :-
    receive(Requests, any, infinity, R, !IO),
    (
        R = message(worker(Reply)),
        spawn(worker, Worker, !IO),
        send(Reply, Worker, !IO),
        serve(Requests, !IO)
    ;
        R = message(quit),
        io.write_string("quitting\n", !IO)
    ;
        R = timeout
    ).

    % Ends when it is told to finish; streams numbers and then waits.
:- pred worker(mailbox(work)::in, io::di, io::uo) is det.

worker(Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(finish)
    ;
        R = message(stream(N, To)),
        list.foldl(send(To), 1 .. N, !IO),
        receive(Self, any, infinity, _, !IO)
    ;
        R = timeout
    ).

:- type seen
    --->    down(down_reason)
    ;       number(int).

:- func down_seen(down_info) = seen.

down_seen(Info) = down(reason(Info)).

:- pred client(string::in, io::di, io::uo) is det.

client(Address, !IO) :-
    connect(Address, Connected, !IO),
    (
        Connected = ok(Node),
        lookup(Node, "server", after(5000), Found, !IO),
        (
            Found = ok(Server),
            watch(Server, !IO)
        ;
            Found = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    ;
        Connected = error(Message),
        io.write_string(Message ++ "\n", !IO)
    ).

:- pred watch(mailbox(server_msg)::in, io::di, io::uo) is det.

watch(Server, !IO) :-
    new_mailbox(Events, !IO),
    new_mailbox(Workers, !IO),

    worker(Server, Workers, A, !IO),
    monitor(A, Events, down_seen, _, !IO),
    send(A, finish, !IO),
    show("ends", Events, !IO),

    worker(Server, Workers, B, !IO),
    monitor(B, Events, down_seen, _, !IO),
    stop(B, !IO),
    show("stopped", Events, !IO),

    worker(Server, Workers, C, !IO),
    monitor(C, Events, down_seen, _, !IO),
    send(C, finish, !IO),
    show("ends", Events, !IO),
    monitor(C, Events, down_seen, _, !IO),
    show("had ended", Events, !IO),

    worker(Server, Workers, D, !IO),
    monitor(D, Events, down_seen, RefD, !IO),
    demonitor(RefD, !IO),
    send(D, finish, !IO),
    receive(Events, any, after(500), E, !IO),
    io.write_string("demonitored: ", !IO),
    io.write_line(E, !IO),

    worker(Server, Workers, F, !IO),
    monitor(F, Events, down_seen, _, !IO),
    new_mailbox(Numbers, !IO),
    send(F, stream(5, Numbers), !IO),
    Both = selecting(Numbers, (func(N) = number(N)),
        selecting(Events, (func(Seen) = Seen), new_selector)),
    stream(Both, 5, !IO),
    send(Server, quit, !IO),
    show("the server quit", Events, !IO),
    monitor(F, Events, down_seen, _, !IO),
    show("watched after the loss", Events, !IO).

:- pred worker(mailbox(server_msg)::in, mailbox(mailbox(work))::in,
    mailbox(work)::out, io::di, io::uo) is det.

worker(Server, Workers, Worker, !IO) :-
    send(Server, worker(Workers), !IO),
    receive(Workers, any, after(5000), R, !IO),
    ( if R = message(W) then
        Worker = W
    else
        unexpected($pred, "no worker")
    ).

:- pred show(string::in, mailbox(seen)::in, io::di, io::uo) is det.

show(What, Events, !IO) :-
    receive(Events, any, after(5000), R, !IO),
    io.write_string(What ++ ": ", !IO),
    io.write_line(R, !IO).

:- pred stream(selector(seen)::in, int::in, io::di, io::uo) is det.

stream(Both, N, !IO) :-
    ( if N = 0 then
        true
    else
        receive(Both, after(5000), R, !IO),
        io.write_string("streamed: ", !IO),
        io.write_line(R, !IO),
        stream(Both, N - 1, !IO)
    ).
