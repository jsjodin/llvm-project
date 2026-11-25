%% The translation of run-gpu-record-arrays.m into Erlang, where every
%% process runs on the host, as the test's do when it is run without an
%% argument. An array is {array, List}, and a cell
%% {cell, {point, X, Y}, Count, Hue, Heat} with its sized integers as
%% {suffixed, N, Suffix}.
-module(gpu_record_arrays_translation).
-export([main/1]).
-mode(compile).
-include("mercury_write.hrl").

i32(N) -> {suffixed, N, "i32"}.
u8(N) -> {suffixed, N band 255, "u8"}.

still() -> {cell, {point, i32(0), i32(0)}, u8(0), red, 0.0}.

next(red) -> green;
next(green) -> blue;
next(blue) -> red.

move({cell, {point, {suffixed, X, _}, {suffixed, Y, _}}, {suffixed, C, _}, H,
      T}) ->
    {cell, {point, i32(X + C), i32(Y - C)}, u8(C + 1), next(H), T / 2.0}.

moved({array, Cells}) ->
    {array, lists:reverse([move(C) || C <- Cells])}.

warm(Cells) ->
    [setelement(5, C, erlang:element(5, C) + float(I))
     || {I, C} <- lists:zip(lists:seq(0, length(Cells) - 1), Cells)].

filled(Fill, {array, Cells}) ->
    {array, warm(Cells ++ [Fill, Fill])}.

cell_of(N, I) ->
    Hue = case I rem 3 of 0 -> red; 1 -> green; 2 -> blue end,
    {cell, {point, i32(I), i32(-N)}, u8((I * 37 + N * 11) rem 256), Hue,
     float(I * N)}.

round_of(N) -> {array, [cell_of(N, I) || I <- lists:seq(0, N - 1)]}.

cells({array, Cells}) -> length(Cells).

mover(Reply) ->
    receive
        A ->
            Reply ! moved(A),
            case cells(A) > 0 of true -> mover(Reply); false -> ok end
    end.

filler(Reply) ->
    receive
        {batch, Fill, A} ->
            Reply ! filled(Fill, A),
            case cells(A) > 0 of true -> filler(Reply); false -> ok end
    end.

forwarder(Next) ->
    receive
        A ->
            Next ! moved(A),
            case cells(A) > 0 of true -> forwarder(Next); false -> ok end
    end.

summer(Report) ->
    receive
        {array, []} ->
            ok;
        {array, Cells} ->
            Report ! {totals, length(Cells),
                      lists:sum([X + Y || {cell, {point, {_, X, _}, {_, Y, _}},
                                           _, _, _} <- Cells]),
                      lists:sum([trunc(T) || {cell, _, _, _, T} <- Cells])},
            summer(Report)
    end.

answer() ->
    receive R -> mercury_write_line({message, R}) end.

main(_) ->
    Self = self(),
    M = spawn(fun() -> mover(Self) end),
    lists:foreach(fun(N) -> M ! round_of(N), answer() end, [3, 1, 0]),
    F = spawn(fun() -> filler(Self) end),
    F ! {batch, cell_of(9, 4), round_of(2)},
    answer(),
    F ! {batch, still(), round_of(0)},
    answer(),
    S = spawn(fun() -> summer(Self) end),
    W = spawn(fun() -> forwarder(S) end),
    lists:foreach(fun(N) -> W ! round_of(N), answer() end, [4, 1000]),
    W ! round_of(0).
