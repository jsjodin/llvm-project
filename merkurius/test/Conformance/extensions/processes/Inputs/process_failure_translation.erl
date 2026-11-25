%% The translation of run-process-failure.m into Erlang. A process that
%% raises an exception writes Merkurius' report of the failure, the
%% exception as io.write writes it, and exits with {failed, Text}, which
%% its monitors deliver. The report goes to standard output: the oracle
%% merges the two streams, and Erlang's servers for them are not ordered
%% with each other, where Merkurius writes the report before the monitors
%% fire.
-module(process_failure_translation).
-export([main/1]).
-mode(compile).
-include("mercury_write.hrl").

throw_exception(Text) ->
    io:format("Uncaught Merkurius exception:~n~s~n", [Text]),
    exit({failed, Text}).

thrower() ->
    receive
        N when N > 0 ->
            throw_exception(mercury_string(
                list_to_binary("too big: " ++ integer_to_list(N))));
        _ -> ok
    end.

doubler() ->
    receive
        {double, N, _} when N > 10 -> throw_exception(mercury_string(N));
        {double, N, {Pid, Tag}} -> Pid ! {Tag, 2 * N}, doubler()
    end.

show_reason({failed, Text}) -> io:format("failed: ~s~n", [Text]);
show_reason(Reason) -> mercury_write_line(Reason).

request(Server, N) ->
    Ref = monitor(process, Server),
    Tag = make_ref(),
    Server ! {double, N, {self(), Tag}},
    receive
        {Tag, Reply} -> demonitor(Ref, [flush]), {reply, Reply};
        {'DOWN', Ref, process, _, Reason} -> {down, Reason}
    end.

main(_) ->
    T = spawn(fun thrower/0),
    monitor(process, T),
    T ! 5,
    receive {'DOWN', _, process, _, Reason} -> show_reason(Reason) end,
    io:format("main goes on~n"),
    S = spawn(fun doubler/0),
    mercury_write_line(request(S, 4)),
    case request(S, 40) of
        {down, R} -> io:format("down, "), show_reason(R);
        Other -> mercury_write_line(Other)
    end.
