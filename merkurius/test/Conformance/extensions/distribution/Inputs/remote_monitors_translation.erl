%% The translation of run-remote-monitors.m into distributed Erlang. The
%% client's mailboxes are its one Erlang mailbox, a monitor's message a
%% DOWN message, and stop an exit with kill.
-module(remote_monitors_translation).
-export([main/1]).
-mode(compile).
-include("nodes.hrl").
-include("../../gpu-processes/Inputs/mercury_write.hrl").

main(["server"]) ->
    listen_node("server"),
    register(server, self()),
    serve();
main(["client", Address]) ->
    start_node("client"),
    Node = connect_node(Address),
    {ok, Server} = lookup_name(Node, server),
    watch(Server).

serve() ->
    receive
        {worker, Reply} ->
            Reply ! {worker, spawn(fun worker/0)},
            serve();
        quit ->
            io:format("quitting~n")
    end.

worker() ->
    receive
        finish -> ok;
        {stream, N, To} ->
            lists:foreach(fun(I) -> To ! {number, I} end, lists:seq(1, N)),
            receive _ -> ok end
    end.

worker(Server) ->
    Server ! {worker, self()},
    receive {worker, W} -> W after 5000 -> error(no_worker) end.

watch(Server) ->
    A = worker(Server),
    monitor(process, A),
    A ! finish,
    show("ends"),

    B = worker(Server),
    monitor(process, B),
    exit(B, kill),
    show("stopped"),

    C = worker(Server),
    monitor(process, C),
    C ! finish,
    show("ends"),
    monitor(process, C),
    show("had ended"),

    D = worker(Server),
    RefD = monitor(process, D),
    demonitor(RefD),
    D ! finish,
    io:format("demonitored: "),
    receive
        {'DOWN', _, process, _, R} -> mercury_write_line({message, {down, R}})
    after 500 -> io:format("timeout~n")
    end,

    F = worker(Server),
    monitor(process, F),
    F ! {stream, 5, self()},
    lists:foreach(fun(_) -> io:format("streamed: "), show_next() end,
                  lists:seq(1, 5)),
    Server ! quit,
    show("the server quit"),
    monitor(process, F),
    show("watched after the loss").

show(What) ->
    io:format("~s: ", [What]),
    show_next().

show_next() ->
    receive
        {'DOWN', _, process, _, R} -> mercury_write_line({message, {down, R}});
        {number, N} -> mercury_write_line({message, {number, N}})
    after 5000 -> io:format("timeout~n")
    end.
