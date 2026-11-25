%% The translation of run-remote-requests.m into distributed Erlang: a
%% registered process serves the requests, and a value is the Erlang term
%% that mercury_write.hrl writes as the Mercury value.
-module(remote_requests_translation).
-export([main/1]).
-mode(compile).
-include("nodes.hrl").
-include("../../gpu-processes/Inputs/mercury_write.hrl").

main(["server"]) ->
    listen_node("server"),
    register(echo, self()),
    serve(0);
main(["client", Address]) ->
    start_node("client"),
    Server = connect_node(Address),
    io:format("connected~n"),
    io:format("the server is another node: ~s~n",
              [yes_no(Server =/= node())]),
    {ok, Echo} = lookup_name(Server, echo),
    io:format("echo is on the server: ~s~n", [yes_no(node(Echo) =:= Server)]),
    requests(Echo).

serve(Served) ->
    receive
        {echo, Value, Reply} ->
            Reply ! Value,
            serve(Served + 1);
        {sum, N, Ints, Reply} ->
            Reply ! {pair, {int, N}, {int, lists:sum(Ints)}},
            serve(Served + 1);
        stop ->
            io:format("served ~w requests~n", [Served])
    end.

requests(Echo) ->
    Values = [
        {int, 42},
        {int, -9223372036854775808},
        {text, <<"">>},
        {text, <<"a string with \"quotes\" and a\nnewline">>},
        {floats, [0.5, -1.25, 100.0]},
        {pair, {int, 1}, {pair, {text, <<"two">>}, none}},
        {values, [none, {int, 3}, {values, []}, {values, [{text, <<"x">>}]}]},
        {values, [{int, I * I} || I <- lists:seq(1, 12)]}
    ],
    lists:foreach(fun(V) -> echo(Echo, V) end, Values),
    lists:foreach(fun(N) -> Echo ! {sum, N, lists:seq(1, N), self()} end,
                  lists:seq(1, 1000)),
    InOrder = lists:all(fun(N) ->
                            receive
                                {pair, {int, N}, {int, S}} ->
                                    S =:= N * (N + 1) div 2;
                                _ -> false
                            after 10000 -> false
                            end
                        end, lists:seq(1, 1000)),
    io:format("1000 pipelined replies in order: ~s~n", [yes_no(InOrder)]),
    Other = spawn_link(fun() ->
                           receive
                               M -> mercury_write_line({message, M})
                           after 10000 -> io:format("timeout~n")
                           end
                       end),
    Echo ! {echo, {text, <<"to a third mailbox">>}, Other},
    Ref = monitor(process, Other),
    receive {'DOWN', Ref, process, Other, _} -> ok end,
    Echo ! stop.

echo(Echo, Value) ->
    Echo ! {echo, Value, self()},
    receive
        Value -> io:format("echoed: ");
        _ -> io:format("changed: ")
    after 10000 -> io:format("timeout: ")
    end,
    mercury_write_line({message, Value}).

yes_no(true) -> "yes";
yes_no(false) -> "no".
