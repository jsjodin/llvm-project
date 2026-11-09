% What gpu_to_gpu.m needs but does not want to show: starting a node, finding
% the other node, and the messages printed along the way.
%
%   gpu_to_gpu b [ADDRESS]    node B: listens, 127.0.0.1:0 by default
%   gpu_to_gpu a HOST:PORT    node A: connects to node B
%
% GPU code cannot print, so each node's host prints what its region is about to
% do and what it gave back.

:- module demo_node.
:- interface.
:- import_module io, process.

    % The two mailboxes of the demo: one on node B that takes Requests, and
    % one on node A that takes Replies. A node can send to either.
:- type links(Request, Reply)
    --->    links(mailbox(Request), mailbox(Reply)).

:- type node(Request, Reply)
    --->    node_a(links(Request, Reply))
    ;       node_b(links(Request, Reply))
    ;       not_started.

    % Starts this node from the command line and connects it to the other.
:- pred start(node(Request, Reply)::out, io::di, io::uo) is det.

:- pred report_sending(int::in, int::in, io::di, io::uo) is det.
:- pred report_sum(int::in, int::in, io::di, io::uo) is det.
:- pred report_waiting(int::in, io::di, io::uo) is det.
:- pred report_received(int::in, int::in, io::di, io::uo) is det.

:- implementation.
:- import_module int, list, maybe, string.

:- func wait = timeout.
wait = after(10000).

start(Node, !IO) :-
    io.command_line_arguments(Arguments, !IO),
    ( if Arguments = ["a", Address] then
        start_a(Address, Node, !IO)
    else if Arguments = ["b", Address] then
        start_b(Address, Node, !IO)
    else if Arguments = ["b"] then
        start_b("127.0.0.1:0", Node, !IO)
    else
        fail_with("usage: gpu_to_gpu b [ADDRESS] | a HOST:PORT", Node, !IO)
    ).

    % Node B offers two mailboxes by name: the one for requests, and one on
    % which node A tells it where to send replies.
:- pred start_b(string::in, node(Request, Reply)::out, io::di, io::uo) is det.
start_b(Address, Node, !IO) :-
    new_mailbox(ToB, !IO),
    new_mailbox(Hello, !IO),
    register("requests", ToB, _, !IO),
    register("hello", Hello, _, !IO),
    listen(Address, Listening, !IO),
    (
        Listening = ok(Port),
        say("B", "listening on port " ++ int_to_string(Port), !IO),
        receive(Hello, any, wait, Got, !IO),
        (
            Got = yes(ToA),
            say("B", "node A has connected", !IO),
            Node = node_b(links(ToB, ToA))
        ;
            Got = no,
            fail_with("node A did not connect", Node, !IO)
        )
    ;
        Listening = error(Error),
        fail_with("listen: " ++ Error, Node, !IO)
    ).

:- pred start_a(string::in, node(Request, Reply)::out, io::di, io::uo) is det.
start_a(Address, Node, !IO) :-
    say("A", "connecting to node B at " ++ Address, !IO),
    connect(Address, Connected, !IO),
    (
        Connected = ok(B),
        lookup(B, "requests", wait, FoundRequests, !IO),
        lookup(B, "hello", wait, FoundHello, !IO),
        ( if
            FoundRequests = ok(ToB),
            FoundHello = ok(Hello)
        then
            new_mailbox(ToA, !IO),
            send(Hello, ToA, !IO),
            say("A", "found node B's mailboxes", !IO),
            Node = node_a(links(ToB, ToA))
        else
            fail_with("node B's mailboxes were not found", Node, !IO)
        )
    ;
        Connected = error(Error),
        fail_with("connect: " ++ Error, Node, !IO)
    ).

%---------------------------------------------------------------------------%

report_sending(Size, Round, !IO) :-
    say("A", "round " ++ int_to_string(Round) ++ ": GPU A sends " ++
        int_to_string(Size) ++ " numbers, each " ++ int_to_string(Round), !IO).

report_sum(Sum, Expected, !IO) :-
    ( if Sum = Expected then
        say("A", "GPU A got the sum " ++ int_to_string(Sum) ++ ", correct", !IO)
    else
        fail_with("GPU A got the sum " ++ int_to_string(Sum) ++ ", expected " ++
            int_to_string(Expected), _ : node(int, int), !IO)
    ).

report_waiting(Round, !IO) :-
    say("B", "round " ++ int_to_string(Round) ++
        ": GPU B waits for an array", !IO).

report_received(Size, Sum, !IO) :-
    ( if Size > 0 then
        say("B", "GPU B got " ++ int_to_string(Size) ++
            " numbers and sent back their sum, " ++ int_to_string(Sum), !IO)
    else
        fail_with("GPU B got no array", _ : node(int, int), !IO)
    ).

:- pred say(string::in, string::in, io::di, io::uo) is det.
say(Name, Text, !IO) :-
    io.write_string("[node " ++ Name ++ "] " ++ Text ++ "\n", !IO),
    io.flush_output(!IO).

:- pred fail_with(string::in, node(Request, Reply)::out, io::di, io::uo)
    is det.
fail_with(Text, not_started, !IO) :-
    io.write_string("error: " ++ Text ++ "\n", !IO),
    io.flush_output(!IO),
    io.set_exit_status(1, !IO).
