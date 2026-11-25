%% The translation of run-remote-calls.m into distributed Erlang. A reply
%% mailbox is {Pid, Ref}, a reply {Ref, Value}, so that a reply that comes
%% after its request gave up matches nothing, as it goes nowhere in
%% Merkurius. request is the design's: a monitor, a send and a receive.
-module(remote_calls_translation).
-export([main/1]).
-mode(compile).
-include("nodes.hrl").
-include("../../gpu-processes/Inputs/mercury_write.hrl").

main(["server"]) ->
    listen_node("server"),
    register(calls, self()),
    serve();
main(["client", Address]) ->
    start_node("client"),
    Node = connect_node(Address),
    {ok, Calls} = lookup_name(Node, calls),
    calls(Calls).

reply({Pid, Ref}, Value) -> Pid ! {Ref, Value}.

serve() ->
    receive
        {add, X, Y, Reply} ->
            reply(Reply, X + Y),
            serve();
        {slow, Milliseconds, Reply} ->
            timer:sleep(Milliseconds),
            reply(Reply, Milliseconds),
            serve();
        {quit, _} ->
            io:format("quitting without a reply~n")
    end.

send_request(Server, MakeRequest) ->
    Reply = {self(), make_ref()},
    Server ! MakeRequest(Reply),
    Reply.

request(Server, MakeRequest, Timeout) ->
    Monitor = monitor(process, Server),
    {_, Ref} = send_request(Server, MakeRequest),
    receive
        {Ref, Value} ->
            demonitor(Monitor, [flush]),
            {reply, Value};
        {'DOWN', Monitor, process, _, Reason} ->
            {down, Reason}
    after Timeout ->
        demonitor(Monitor, [flush]),
        timeout
    end.

calls(Calls) ->
    R1 = request(Calls, fun(Reply) -> {add, 2, 3, Reply} end, 5000),
    io:format("add: "),
    mercury_write_line(R1),
    R2 = request(Calls, fun(Reply) -> {slow, 1000, Reply} end, 100),
    io:format("slow, waiting 100 ms: "),
    mercury_write_line(R2),
    {_, Ref} = send_request(Calls, fun(Reply) -> {add, 40, 2, Reply} end),
    io:format("sent a request~n"),
    receive
        {Ref, Value} -> io:format("collected: "),
                        mercury_write_line({message, Value})
    after 5000 -> io:format("timeout~n")
    end,
    R4 = request(Calls, fun(Reply) -> {quit, Reply} end, 5000),
    io:format("quit: "),
    mercury_write_line(R4).
