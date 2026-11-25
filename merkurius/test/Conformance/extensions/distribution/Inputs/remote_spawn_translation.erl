%% The translation of nyi-remote-spawn.m into distributed Erlang: the
%% client has the server's node start the counter, by a call on that node,
%% which runs the same module, as spawn_counter_on would.
-module(remote_spawn_translation).
-export([main/1, start_counter/1]).
-mode(compile).
-include("nodes.hrl").
-include("../../gpu-processes/Inputs/mercury_write.hrl").

counter(N) ->
    receive
        {add, K} -> counter(N + K);
        {get, Reply} -> Reply ! N, counter(N);
        stop -> io:format("the counter stops at ~w~n", [N])
    end.

%% Runs on the server's node; the counter's output goes to that node's.
start_counter(N) ->
    spawn(fun() -> group_leader(whereis(user), self()), counter(N) end).

main(["server"]) ->
    listen_node("server"),
    register(quit, self()),
    receive _ -> ok end;
main(["client", Address]) ->
    start_node("client"),
    Node = connect_node(Address),
    Counter = rpc:call(Node, ?MODULE, start_counter, [5]),
    io:format("the counter is on the server: ~s~n",
              [case node(Counter) =:= Node of true -> "yes"; false -> "no"
               end]),
    Counter ! {add, 2},
    Counter ! {get, self()},
    receive N -> mercury_write_line({message, N}) after 5000 -> ok end,
    Counter ! stop,
    {ok, Quit} = lookup_name(Node, quit),
    Quit ! 0.
