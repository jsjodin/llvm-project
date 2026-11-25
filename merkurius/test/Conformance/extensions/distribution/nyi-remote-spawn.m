% DESIGN: Distribution and wire format / Remote spawn
% A process kind is spawned on another node by its name, with
% spawn_Kind_on from its declaration, which returns ok(Mailbox) or an
% error. The node that runs it contains the definition; here both nodes
% run the same program. The translation into distributed Erlang has the
% server's node start the counter, by a call on that node, and gives the
% output the program must print once remote spawn is built.
% MERKC-FLAGS: -fprocess-declarations
% TRANSLATION: Inputs/remote_spawn_translation.erl
% NODES: server client
%
% RUN: not merkc -fprocess-declarations %s -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: spawning a process kind on another node (spawn_counter_on/5)
:- module nyi_remote_spawn.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, process, string.

:- type counter_msg
    --->    add(int)
    ;       get(mailbox(int))
    ;       stop.

:- process counter(int) receives counter_msg.

counter(N, Self, !IO) :-
    receive(Self, any, infinity, R, !IO),
    (
        R = message(add(K)),
        counter(N + K, Self, !IO)
    ;
        R = message(get(Reply)),
        send(Reply, N, !IO),
        counter(N, Self, !IO)
    ;
        R = message(stop),
        io.format("the counter stops at %d\n", [i(N)], !IO)
    ;
        R = timeout
    ).

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    ( if Args = ["server"] then
        listen("127.0.0.1:0", Listened, !IO),
        (
            Listened = ok(Port),
            io.format("listening 127.0.0.1:%d\n", [i(Port)], !IO),
            io.flush_output(!IO),
            new_mailbox(Quit, !IO),
            register("quit", Quit, _, !IO),
            receive(Quit : mailbox(int), any, infinity, _, !IO)
        ;
            Listened = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    else if Args = ["client", Address] then
        connect(Address, Connected, !IO),
        (
            Connected = ok(Node),
            spawn_counter_on(Node, 5, Spawned, !IO),
            (
                Spawned = ok(Counter),
                io.format("the counter is on the server: %s\n",
                    [s(if node_of(Counter) = Node then "yes" else "no")], !IO),
                send(Counter, add(2), !IO),
                new_mailbox(Replies, !IO),
                send(Counter, get(Replies), !IO),
                receive(Replies, any, after(5000), R, !IO),
                io.write_line(R, !IO),
                send(Counter, stop, !IO),
                lookup(Node, "quit", after(5000), Found, !IO),
                ( if Found = ok(Quit) then
                    send(Quit, 0, !IO)
                else
                    true
                )
            ;
                Spawned = error(_),
                io.write_string("cannot spawn\n", !IO)
            )
        ;
            Connected = error(Message),
            io.write_string(Message ++ "\n", !IO)
        )
    else
        io.write_string("usage: server | client ADDRESS\n", !IO)
    ).
