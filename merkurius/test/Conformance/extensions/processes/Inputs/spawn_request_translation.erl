%% The translation of run-spawn-request.m into Erlang. A process's mailbox
%% is its pid; a reply mailbox is {Pid, Tag}, its messages tagged so, and
%% request is gen_server:call's pattern: monitor the server, send, and wait
%% for the reply, its DOWN message or the timeout.
-module(spawn_request_translation).
-export([main/1]).
-mode(compile).
-include("../../gpu-processes/Inputs/mercury_write.hrl").

counter(N) ->
    receive
        {increment, K} -> counter(N + K);
        {get, {Pid, Tag}} -> Pid ! {Tag, N}, counter(N);
        stop -> ok
    end.

silent() ->
    receive _ -> silent() end.

timeout_ms(infinity) -> infinity;
timeout_ms({'after', Ms}) -> Ms.

request(Server, Timeout) ->
    Ref = monitor(process, Server),
    Tag = make_ref(),
    Server ! {get, {self(), Tag}},
    receive
        {Tag, N} -> demonitor(Ref, [flush]), {reply, N};
        {'DOWN', Ref, process, _, Reason} -> {down, Reason}
    after timeout_ms(Timeout) -> demonitor(Ref, [flush]), timeout
    end.

send_request(Server) ->
    Tag = make_ref(),
    Server ! {get, {self(), Tag}},
    Tag.

show_down(Label) ->
    io:format("~s: ", [Label]),
    receive
        {'DOWN', _, process, _, Reason} -> mercury_write_line(Reason)
    end.

main(_) ->
    C = spawn(fun() -> counter(0) end),
    C ! {increment, 1},
    C ! {increment, 2},
    C ! {increment, 3},
    mercury_write_line(request(C, infinity)),
    C ! {increment, 4},
    Tag = send_request(C),
    Got = receive {Tag, N} -> {message, N} end,
    mercury_write_line(Got),
    monitor(process, C),
    C ! stop,
    show_down("returned"),
    monitor(process, C),
    show_down("ended already"),
    mercury_write_line(request(C, infinity)),
    S = spawn(fun silent/0),
    mercury_write_line(request(S, {'after', 50})),
    monitor(process, S),
    exit(S, kill),
    show_down("stopped").
