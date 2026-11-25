%% The translation of run-gpu-team-loops.m into Erlang: the solver is a
%% process, and its loops are sequential, as each of the Mercury program's
%% loops means the same in any order of its iterations.
-module(gpu_team_loops_translation).
-export([main/1]).
-mode(compile).
-include("mercury_write.hrl").

-define(MAX_INT, 9223372036854775807).
-define(MIN_INT, -9223372036854775808).

solver(Reply) ->
    receive
        {tiles, T, S} ->
            Tiles = [tile(J, S) || J <- lists:seq(0, T - 1)],
            Sums = [Sum || {Sum, _, _} <- Tiles],
            Last = case T > 0 of true -> lists:last(Sums); false -> -1 end,
            Reply ! {tiled, checksum(Sums),
                     lists:foldl(fun erlang:min/2, ?MAX_INT,
                                 [Lo || {_, Lo, _} <- Tiles]),
                     lists:foldl(fun erlang:max/2, ?MIN_INT,
                                 [Hi || {_, _, Hi} <- Tiles]),
                     Last},
            solver(Reply);
        {smooth, N, Steps} ->
            Start = [(I * I) rem 1000 || I <- lists:seq(0, N - 1)],
            End = smooth(Steps, Start),
            Reply ! {smoothed, checksum(End), lists:nth(N div 2 + 1, End)},
            solver(Reply);
        {big, N} ->
            Sum = lists:sum([I rem 1000 || I <- lists:seq(0, N - 1)]),
            Reply ! {summed, Sum},
            solver(Reply);
        stop ->
            ok
    end.

tile(J, S) ->
    Vs = [(J * 7919 + K * 104729) rem 10007 || K <- lists:seq(0, S - 1)],
    {lists:sum(Vs), lists:foldl(fun erlang:min/2, ?MAX_INT, Vs),
     lists:foldl(fun erlang:max/2, ?MIN_INT, Vs)}.

smooth(0, A) -> A;
smooth(Steps, A) -> smooth(Steps - 1, step(A)).

step([First | Rest] = A) ->
    [First | inner(A, Rest)].

inner([X, Y, Z | _] = [_ | More], _) ->
    [(X + Y + Z) rem 1000003 | inner(More, ignored)];
inner([_, Last], _) ->
    [Last];
inner(_, _) ->
    [].

checksum(A) ->
    Add = fun(X, {I, P}) -> {I + 1, (P + X * I) rem 1000000007} end,
    {_, Sum} = lists:foldl(Add, {1, 0}, A),
    Sum.

ask(Solver, Job) ->
    Solver ! Job,
    receive R ->
        mercury_write(Job),
        io:put_chars(": "),
        mercury_write_line({message, R})
    end.

main(_) ->
    Self = self(),
    S = spawn(fun() -> solver(Self) end),
    ask(S, {tiles, 1, 1}),
    ask(S, {tiles, 5, 300}),
    ask(S, {tiles, 500, 70}),
    ask(S, {tiles, 0, 10}),
    ask(S, {smooth, 10, 0}),
    ask(S, {smooth, 1000, 20}),
    ask(S, {big, 2000000}),
    S ! stop.
