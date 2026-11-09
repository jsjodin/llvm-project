% What gpu_to_gpu.m needs but does not want to show: starting a node, finding
% the other node, and the messages printed along the way.
%
%   gpu_to_gpu b [ADDRESS]    node B: listens, 127.0.0.1:0 by default
%   gpu_to_gpu a HOST:PORT    node A: connects to node B, waiting for it

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

:- pred report_round(string::in, int::in, io::di, io::uo) is det.

    % These say so, and fail the program, if a GPU's region went wrong.
:- pred check_sum(int::in, int::in, io::di, io::uo) is det.
:- pred report_timeout(string::in, io::di, io::uo) is det.

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
            Got = message(ToA),
            say("B", "node A has connected", !IO),
            Node = node_b(links(ToB, ToA))
        ;
            Got = timeout,
            fail_with("node A did not connect", Node, !IO)
        )
    ;
        Listening = error(Error),
        fail_with("listen: " ++ Error, Node, !IO)
    ).

:- pred start_a(string::in, node(Request, Reply)::out, io::di, io::uo) is det.
start_a(Address, Node, !IO) :-
    say("A", "connecting to node B at " ++ Address, !IO),
    connect_retrying(Address, connect_tries, Connected, !IO),
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

:- func connect_tries = int.
connect_tries = 50.

    % Node B may not be listening yet: tries each 100 ms.
:- pred connect_retrying(string::in, int::in, maybe_error(node)::out,
    io::di, io::uo) is det.
connect_retrying(Address, Tries, Connected, !IO) :-
    connect(Address, Result, !IO),
    ( if Result = error(_), Tries > 1 then
        pause(!IO),
        connect_retrying(Address, Tries - 1, Connected, !IO)
    else
        Connected = Result
    ).

:- pragma foreign_decl("C", local, "#include <unistd.h>").

:- pred pause(io::di, io::uo) is det.
:- pragma foreign_proc("C", pause(_IO0::di, _IO::uo),
    [will_not_call_mercury, promise_pure],
"
    usleep(100000);
").

%---------------------------------------------------------------------------%

report_round(Name, Round, !IO) :-
    say(Name, "round " ++ int_to_string(Round), !IO).

check_sum(Sum, Expected, !IO) :-
    ( if Sum = Expected then
        true
    else
        fail_with("GPU A got the sum " ++ int_to_string(Sum) ++ ", expected " ++
            int_to_string(Expected), _ : node(int, int), !IO)
    ).

report_timeout(Waiting, !IO) :-
    fail_with(Waiting ++ ": nothing arrived in time", _ : node(int, int), !IO).

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
