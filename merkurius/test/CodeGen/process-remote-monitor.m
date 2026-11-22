% Monitors and stops across two and three nodes over loopback (docs/INPROGRESS.md,
% Failure, step 2), for which mmc cannot be the oracle, as it has no process
% module. A driver runs the program in its roles, one process per node; see
% Inputs/two-nodes.py.
%
% - monitors: a client watches processes of the server, which end normally,
%   fail, are stopped from the client, or have ended before the monitor; a
%   demonitor stops the message; a request to the server's process gets its
%   reply, and down once that process is stopped. When the server quits, a
%   monitor of its process says noconnection, as does one made afterwards,
%   and a stop of it goes nowhere.
% - fence: the driver stops the server with SIGSTOP. The client's monitor
%   says noconnection once the connection is lost; once the server runs
%   again, its monitor of the client's process says noconnection too, and
%   the client cannot connect to it again, as a node that was lost is
%   fenced off.
% - in-flight: the server's process streams numbers to the client, which
%   watches it, and the driver kills the server with SIGKILL. The numbers
%   that arrive come in order with none missing, then noconnection, and
%   nothing after it: delivery is at most once, and a lost node's messages
%   all come before its noconnection.
% - three: a client watches a process on each of two servers, and the
%   driver kills one. Only that one's monitor says noconnection; the other
%   server still answers, and its process is stopped from the client.
% - partition: the client runs with the fault injector's partitions
%   (MERK_FAULTS, docs/INPROGRESS.md, Failure, step 3), which silence its
%   connection to the server. The ticks lose it, the client's monitor says
%   noconnection, and the fence refuses a new connection.
%
% RUN: merkc -O0 %s -o %t0 && %python %S/Inputs/two-nodes.py %t0 monitors fence in-flight three partition | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %python %S/Inputs/two-nodes.py %t2 monitors fence in-flight three partition | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
%
% Every message is read or freed, on both nodes, downs from the other node
% among them.
% RUN: env MERK_CHECK_MESSAGES=1 %python %S/Inputs/two-nodes.py %t2 monitors | FileCheck %s --check-prefix=MESSAGES --match-full-lines --strict-whitespace
%
% OUTPUT:monitors:
% OUTPUT-NEXT:client: connected
% OUTPUT-NEXT:client: finish: normal
% OUTPUT-NEXT:client: divide: domain_error("int.\'//\': division by zero")
% OUTPUT-NEXT:client: stopped: killed
% OUTPUT-NEXT:client: ended: noproc
% OUTPUT-NEXT:client: after a demonitor: nothing
% OUTPUT-NEXT:client: then: {{normal|noproc}}
% OUTPUT-NEXT:client: request: reply 42
% OUTPUT-NEXT:client: asked: killed
% OUTPUT-NEXT:client: request after the stop: down noproc
% OUTPUT-NEXT:client: hanging: noconnection
% OUTPUT-NEXT:client: after the loss: noconnection
% OUTPUT-NEXT:server: listening
% OUTPUT-NEXT:server exited with 0: 'Uncaught Merkurius exception:\ndomain_error("int.\\\'//\\\': division by zero")\n'
% OUTPUT-NEXT:server: quitting
% OUTPUT-NEXT:fence:
% OUTPUT-NEXT:client: ready
% OUTPUT-NEXT:client: hanging: noconnection
% OUTPUT-NEXT:client: connect again: 127.0.0.1:PORT: the node was lost before, and is not connected to again: it must start again
% OUTPUT-NEXT:client: after the loss: noconnection
% OUTPUT-NEXT:server: listening
% OUTPUT-NEXT:server: watching
% OUTPUT-NEXT:server: the client's process: noconnection
% OUTPUT-NEXT:in-flight:
% OUTPUT-NEXT:client: streaming
% OUTPUT-NEXT:client: then: noconnection
% OUTPUT-NEXT:client: in order: yes
% OUTPUT-NEXT:client: after it: nothing
% OUTPUT-NEXT:server: killed
% OUTPUT-NEXT:three:
% OUTPUT-NEXT:client: ready
% OUTPUT-NEXT:client: b: noconnection
% OUTPUT-NEXT:client: c after b's loss: reply 42
% OUTPUT-NEXT:client: c: killed
% OUTPUT-NEXT:client: c: quitting
% OUTPUT-NEXT:b: killed
% OUTPUT-NEXT:c: listening
% OUTPUT-NEXT:c: quitting
% OUTPUT-NEXT:partition:
% OUTPUT-NEXT:client: hanging: noconnection
% OUTPUT-NEXT:client: connect again: 127.0.0.1:PORT: the node was lost before, and is not connected to again: it must start again
% OUTPUT-NEXT:schedule: # seed=1
% OUTPUT-NEXT:schedule: MS partition 1
% OUTPUT-NOT:{{.}}
%
% MESSAGES:monitors:
% MESSAGES-NEXT:client exited with 0: 'Merkurius runtime: 0 messages are neither read nor freed\n'
% MESSAGES-NEXT:client: connected
% MESSAGES:server exited with 0: 'Uncaught Merkurius exception:\ndomain_error("int.\\\'//\\\': division by zero")\nMerkurius runtime: 0 messages are neither read nor freed\n'

:- module process_remote_monitor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module bool, int, list, maybe, process, require, string.

:- type order
    --->    finish
    ;       divide(int)
    ;       hang
    ;       ask(mailbox(int))
    ;       stream(mailbox(int)).

:- type command
    --->    target(mailbox(mailbox(order)))
    ;       watch(mailbox(int), mailbox(string))
    ;       quit(mailbox(string)).

:- type input
    --->    command(command)
    ;       went(down_reason).

:- type tracked
    --->    number(int)
    ;       down(down_info).

main(!IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["server"] then
        server(!IO)
    else if Arguments = ["client", Port] then
        client(Port, !IO)
    else if Arguments = ["fenced", Port] then
        fenced(Port, !IO)
    else if Arguments = ["in-flight", Port] then
        in_flight(Port, !IO)
    else if Arguments = ["partitioned", Port] then
        partitioned(Port, !IO)
    else if Arguments = ["three", PortB, PortC] then
        three(PortB, PortC, !IO)
    else
        io.write_string("usage: server | client PORT | fenced PORT | " ++
            "in-flight PORT | three PORT PORT\n", !IO)
    ).

%---------------------------------------------------------------------------%
% The server: makes processes for the client, and watches one of its.

:- pred server(io::di, io::uo) is det.

server(!IO) :-
    new_mailbox(Commands : mailbox(command), !IO),
    new_mailbox(Downs : mailbox(down_reason), !IO),
    register("commands", Commands, _, !IO),
    listen("127.0.0.1:0", Listening, !IO),
    (
        Listening = ok(Port),
        io.write_string("listening " ++ int_to_string(Port) ++ "\n", !IO),
        io.flush_output(!IO),
        serve(Commands, Downs, !IO)
    ;
        Listening = error(Error),
        io.write_string("listen: " ++ Error ++ "\n", !IO)
    ),
    close_mailbox(Commands, !IO),
    close_mailbox(Downs, !IO).

:- pred serve(mailbox(command)::in, mailbox(down_reason)::in,
    io::di, io::uo) is det.

serve(Commands, Downs, !IO) :-
    Selector = selecting(Commands, (func(C) = command(C)),
        selecting(Downs, (func(D) = went(D)), new_selector)),
    receive(Selector, infinity, Got, !IO),
    (
        Got = message(command(target(ReplyTo))),
        spawn(target, Target, !IO),
        send(ReplyTo, Target, !IO),
        serve(Commands, Downs, !IO)
    ;
        Got = message(command(watch(Watched, ReplyTo))),
        monitor(Watched, Downs, reason, _, !IO),
        io.write_string("watching\n", !IO),
        io.flush_output(!IO),
        send(ReplyTo, "watching", !IO),
        serve(Commands, Downs, !IO)
    ;
        Got = message(command(quit(ReplyTo))),
        send(ReplyTo, "quitting", !IO),
        io.write_string("quitting\n", !IO)
    ;
        Got = message(went(Reason)),
        io.write_string("the client's process: " ++ reason_text(Reason) ++
            "\n", !IO),
        % Up until the client has tried to connect again.
        io.read_line_as_string(_, !IO)
    ;
        Got = timeout
    ).

:- pred target(mailbox(order)::in, io::di, io::uo) is det.

target(Box, !IO) :-
    receive(Box, any, infinity, Got, !IO),
    (
        Got = message(finish)
    ;
        Got = message(divide(N)),
        io.write_int(100 // N, !IO)
    ;
        Got = message(hang),
        target(Box, !IO)
    ;
        Got = message(ask(ReplyTo)),
        send(ReplyTo, 42, !IO),
        target(Box, !IO)
    ;
        Got = message(stream(To)),
        stream_from(1, To, Box, !IO)
    ;
        Got = timeout
    ).

    % Sends 1, 2, ... for ever, with a pause after each hundred, so that
    % the receiver keeps up.
    %
:- pred stream_from(int::in, mailbox(int)::in, mailbox(order)::in,
    io::di, io::uo) is det.

stream_from(K, To, Box, !IO) :-
    send(To, K, !IO),
    ( if K mod 100 = 0 then
        receive(Box, any, after(1), _ : received(order), !IO)
    else
        true
    ),
    stream_from(K + 1, To, Box, !IO).

%---------------------------------------------------------------------------%
% The client.

:- pred client(string::in, io::di, io::uo) is det.

client(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Server),
        io.write_string("connected\n", !IO),
        lookup(Server, "commands", after(10000), Found, !IO),
        (
            Found = ok(Commands),
            new_mailbox(Downs, !IO),
            watch_all(Commands, Downs, !IO),
            close_mailbox(Downs, !IO)
        ;
            Found = error(Error),
            io.write_string("lookup: " ++ Error ++ "\n", !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO)
    ).

:- pred watch_all(mailbox(command)::in, mailbox(down_info)::in,
    io::di, io::uo) is det.

watch_all(Commands, Downs, !IO) :-
    % A process that ends by itself.
    new_target(Commands, Finish, !IO),
    watch(Finish, Downs, !IO),
    send(Finish, finish, !IO),
    show_down("finish", Downs, !IO),

    % One that fails.
    new_target(Commands, Divide, !IO),
    watch(Divide, Downs, !IO),
    send(Divide, divide(0), !IO),
    show_down("divide", Downs, !IO),

    % One stopped from here.
    new_target(Commands, Stopped, !IO),
    watch(Stopped, Downs, !IO),
    stop(Stopped, !IO),
    show_down("stopped", Downs, !IO),

    % One that has ended.
    watch(Finish, Downs, !IO),
    show_down("ended", Downs, !IO),

    % A demonitor: the end that follows sends nothing. Whether a monitor
    % made after the order sees the end or finds it has happened is a race.
    new_target(Commands, Unwatched, !IO),
    monitor(Unwatched, Downs, (func(D) = D), Ref, !IO),
    demonitor(Ref, !IO),
    send(Unwatched, finish, !IO),
    show_down_within(1000, "after a demonitor", Downs, !IO),
    watch(Unwatched, Downs, !IO),
    show_down("then", Downs, !IO),

    % Requests, whose monitors are across nodes too.
    new_target(Commands, Asked, !IO),
    request(Asked, (func(R) = ask(R)), after(10000), First, !IO),
    show_reply("request", First, !IO),
    watch(Asked, Downs, !IO),
    stop(Asked, !IO),
    show_down("asked", Downs, !IO),
    request(Asked, (func(R) = ask(R)), after(10000), Second, !IO),
    show_reply("request after the stop", Second, !IO),

    % The server quits, and the connection is lost.
    new_target(Commands, Hanging, !IO),
    watch(Hanging, Downs, !IO),
    send(Hanging, hang, !IO),
    new_mailbox(Replies, !IO),
    send(Commands, quit(Replies), !IO),
    receive(Replies, any, after(10000), _ : received(string), !IO),
    close_mailbox(Replies, !IO),
    show_down("hanging", Downs, !IO),
    watch(Hanging, Downs, !IO),
    show_down("after the loss", Downs, !IO),
    stop(Hanging, !IO).

    % Numbers in flight when the server is killed.
    %
:- pred in_flight(string::in, io::di, io::uo) is det.

in_flight(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    ( if Connected = ok(Server) then
        lookup(Server, "commands", after(10000), Found, !IO),
        ( if Found = ok(Commands) then
            new_target(Commands, Streamer, !IO),
            new_mailbox(Numbers, !IO),
            new_mailbox(Downs, !IO),
            watch(Streamer, Downs, !IO),
            send(Streamer, stream(Numbers), !IO),
            Selector = selecting(Numbers, (func(N) = number(N)),
                selecting(Downs, (func(D) = down(D)), new_selector)),
            follow(Selector, 1, 1000, yes, InOrder0, Next, !IO),
            io.write_string("streaming\n", !IO),
            io.flush_output(!IO),
            follow(Selector, Next, max_int, InOrder0, InOrder, _, !IO),
            io.write_string("in order: " ++ yes_no(InOrder) ++ "\n", !IO),
            receive(Selector, after(500), After, !IO),
            ( if After = timeout then
                io.write_string("after it: nothing\n", !IO)
            else
                io.write_string("after it: something\n", !IO)
            ),
            close_mailbox(Numbers, !IO),
            close_mailbox(Downs, !IO)
        else
            io.write_string("no lookup\n", !IO)
        )
    else
        io.write_string("no connection\n", !IO)
    ).

    % Reads numbers until one past Last has been read, or a down comes,
    % which it shows; whether each was the one after the one before.
    %
:- pred follow(selector(tracked)::in, int::in, int::in, bool::in, bool::out,
    int::out, io::di, io::uo) is det.

follow(Selector, Expected, Last, !InOrder, Next, !IO) :-
    ( if Expected > Last then
        Next = Expected
    else
        receive(Selector, after(10000), Got, !IO),
        (
            Got = message(number(N)),
            ( if N = Expected then true else !:InOrder = no ),
            follow(Selector, N + 1, Last, !InOrder, Next, !IO)
        ;
            Got = message(down(Down)),
            io.write_string("then: " ++ reason_text(reason(Down)) ++ "\n",
                !IO),
            Next = Expected
        ;
            Got = timeout,
            io.write_string("timeout\n", !IO),
            Next = Expected
        )
    ).

:- func yes_no(bool) = string.

yes_no(yes) = "yes".
yes_no(no) = "no".

    % A watch across a connection that the injector silences.
    %
:- pred partitioned(string::in, io::di, io::uo) is det.

partitioned(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    ( if Connected = ok(Server) then
        lookup(Server, "commands", after(10000), Found, !IO),
        ( if Found = ok(Commands) then
            new_target(Commands, Hanging, !IO),
            new_mailbox(Downs, !IO),
            watch(Hanging, Downs, !IO),
            show_down("hanging", Downs, !IO),
            connect("127.0.0.1:" ++ Port, Again, !IO),
            show_connect(Port, Again, !IO),
            close_mailbox(Downs, !IO)
        else
            io.write_string("no lookup\n", !IO)
        )
    else
        io.write_string("no connection\n", !IO)
    ).

    % Two servers, of which the driver kills the first.
    %
:- pred three(string::in, string::in, io::di, io::uo) is det.

three(PortB, PortC, !IO) :-
    connect("127.0.0.1:" ++ PortB, ConnectedB, !IO),
    connect("127.0.0.1:" ++ PortC, ConnectedC, !IO),
    ( if
        ConnectedB = ok(B),
        ConnectedC = ok(C)
    then
        lookup(B, "commands", after(10000), FoundB, !IO),
        lookup(C, "commands", after(10000), FoundC, !IO),
        ( if
            FoundB = ok(CommandsB),
            FoundC = ok(CommandsC)
        then
            new_target(CommandsB, TargetB, !IO),
            new_target(CommandsC, TargetC, !IO),
            new_mailbox(DownsB, !IO),
            new_mailbox(DownsC, !IO),
            watch(TargetB, DownsB, !IO),
            watch(TargetC, DownsC, !IO),
            io.write_string("ready\n", !IO),
            io.flush_output(!IO),
            io.read_line_as_string(_, !IO),
            show_down("b", DownsB, !IO),
            request(TargetC, (func(R) = ask(R)), after(10000), Reply, !IO),
            show_reply("c after b's loss", Reply, !IO),
            stop(TargetC, !IO),
            show_down("c", DownsC, !IO),
            new_mailbox(Replies, !IO),
            send(CommandsC, quit(Replies), !IO),
            receive(Replies, any, after(10000), Quit, !IO),
            ( if Quit = message(Text) then
                io.write_string("c: " ++ Text ++ "\n", !IO)
            else
                io.write_string("c: no answer\n", !IO)
            ),
            close_mailbox(Replies, !IO),
            close_mailbox(DownsB, !IO),
            close_mailbox(DownsC, !IO)
        else
            io.write_string("no lookup\n", !IO)
        )
    else
        io.write_string("no connection\n", !IO)
    ).

    % The fence: the server is stopped while this node watches its process,
    % and lost; then it runs again, and this node cannot connect to it.
    %
:- pred fenced(string::in, io::di, io::uo) is det.

fenced(Port, !IO) :-
    connect("127.0.0.1:" ++ Port, Connected, !IO),
    (
        Connected = ok(Server),
        lookup(Server, "commands", after(10000), Found, !IO),
        (
            Found = ok(Commands),
            new_mailbox(Downs, !IO),
            new_target(Commands, Hanging, !IO),
            watch(Hanging, Downs, !IO),
            new_mailbox(Mine : mailbox(int), !IO),
            new_mailbox(Replies, !IO),
            send(Commands, watch(Mine, Replies), !IO),
            receive(Replies, any, after(10000), _ : received(string), !IO),
            io.write_string("ready\n", !IO),
            io.flush_output(!IO),
            io.read_line_as_string(_, !IO),
            show_down("hanging", Downs, !IO),
            io.flush_output(!IO),
            io.read_line_as_string(_, !IO),
            connect("127.0.0.1:" ++ Port, Again, !IO),
            show_connect(Port, Again, !IO),
            watch(Hanging, Downs, !IO),
            show_down("after the loss", Downs, !IO),
            close_mailbox(Mine, !IO),
            close_mailbox(Replies, !IO),
            close_mailbox(Downs, !IO)
        ;
            Found = error(Error),
            io.write_string("lookup: " ++ Error ++ "\n", !IO)
        )
    ;
        Connected = error(Error),
        io.write_string("connect: " ++ Error ++ "\n", !IO)
    ).

:- pred new_target(mailbox(command)::in, mailbox(order)::out,
    io::di, io::uo) is det.

new_target(Commands, Target, !IO) :-
    request(Commands, (func(R) = target(R)), after(10000), Reply, !IO),
    ( if Reply = reply(Made) then
        Target = Made
    else
        error("new_target: no target")
    ).

:- pred watch(mailbox(order)::in, mailbox(down_info)::in,
    io::di, io::uo) is det.

watch(Target, Downs, !IO) :-
    monitor(Target, Downs, (func(D) = D), _, !IO).

:- pred show_down(string::in, mailbox(down_info)::in, io::di, io::uo) is det.

show_down(Name, Downs, !IO) :-
    % Long enough for the loss of a silent node, after a few tick times.
    show_down_within(10000, Name, Downs, !IO).

:- pred show_down_within(int::in, string::in, mailbox(down_info)::in,
    io::di, io::uo) is det.

show_down_within(Milliseconds, Name, Downs, !IO) :-
    receive(Downs, any, after(Milliseconds), Got, !IO),
    (
        Got = message(Down),
        io.write_string(Name ++ ": " ++ reason_text(reason(Down)) ++ "\n",
            !IO)
    ;
        Got = timeout,
        io.write_string(Name ++ ": nothing\n", !IO)
    ).

:- pred show_reply(string::in, reply(int)::in, io::di, io::uo) is det.

show_reply(Name, reply(N), !IO) :-
    io.format("%s: reply %d\n", [s(Name), i(N)], !IO).
show_reply(Name, timeout, !IO) :-
    io.write_string(Name ++ ": timeout\n", !IO).
show_reply(Name, down(Reason), !IO) :-
    io.write_string(Name ++ ": down " ++ reason_text(Reason) ++ "\n", !IO).

:- pred show_connect(string::in, maybe_error(node)::in, io::di, io::uo) is det.

show_connect(_, ok(_), !IO) :-
    io.write_string("connect again: connected\n", !IO).
show_connect(Port, error(Error), !IO) :-
    io.write_string("connect again: " ++ replace_all(Error, Port, "PORT") ++
        "\n", !IO).
