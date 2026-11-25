%% The translation of run-gpu-kinds.m into Erlang, where every process
%% runs on the host, as the test's do when it is run without an argument.
-module(gpu_kinds_translation).
-export([main/1]).
-mode(compile).
-include("mercury_write.hrl").

counter(N) ->
    receive
        {add, K} -> counter(N + K);
        {urgent, K} -> counter(N + K * 100);
        {get, Reply} -> Reply ! {count, N, N rem 100}, counter(N);
        stop -> ok
    end.

poller(Report) ->
    receive
        _ -> Report ! {polled, <<"a message">>}
    after 0 -> Report ! {polled, <<"timeout">>}
    end.

ponger() ->
    receive
        {ping, N, Back} -> Back ! N * 2 + 1, ponger();
        done -> ok
    end.

pinger(0, N, Ponger, Report) ->
    Report ! {pinged, N},
    Ponger ! done;
pinger(Rounds, N, Ponger, Report) ->
    Ponger ! {ping, N, self()},
    receive M -> pinger(Rounds - 1, M rem 1000, Ponger, Report) end.

worker(0, _) ->
    ok;
worker(K, Report) ->
    receive
        {ints, A} -> Report ! {summed, lists:sum(A)};
        {floats, A} -> Report ! {summed_floats, lists:sum(A)};
        {move, {point, X, Y}, D} -> Report ! {moved, {point, X + D, Y * 2.0}}
    end,
    worker(K - 1, Report).

report() ->
    receive R -> mercury_write_line({message, R}) end.

main(_) ->
    Self = self(),
    C = spawn(fun() -> counter(0) end),
    C ! {add, 1},
    C ! {add, 2},
    C ! {urgent, 3},
    C ! {get, Self},
    report(),
    C ! stop,
    spawn(fun() -> poller(Self) end),
    report(),
    P = spawn(fun ponger/0),
    spawn(fun() -> pinger(50, 0, P, Self) end),
    report(),
    W = spawn(fun() -> worker(3, Self) end),
    W ! {ints, lists:seq(1, 100)},
    report(),
    W ! {floats, [0.5, 1.25, 2.0]},
    report(),
    W ! {move, {point, 4, 1.5}, 3},
    report().
