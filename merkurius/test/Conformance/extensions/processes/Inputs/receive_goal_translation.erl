%% The translation of run-receive-goal.m into Erlang. A process's mailboxes
%% are its one Erlang mailbox, each message tagged with its mailbox: main's
%% a, b, c, setup, replies and notes_back, and the server's notes. A
%% mailbox value is {Pid, Tag}, and the server's own mailbox its pid. A
%% receive goal is Erlang's receive, its arms the clauses, in order.
-module(receive_goal_translation).
-export([main/1]).
-mode(compile).
%% any_first has a clause that cannot match, as the test means.
-compile([nowarn_nomatch]).
-include("mercury_write.hrl").

send({Pid, Tag}, M) ->
    Pid ! {Tag, M}.

server(Count, Notes) ->
    receive
        {notes, {note, Text}} -> server(Count, [Text | Notes]);
        {increment, N} -> server(Count + N, Notes);
        {get, Id, ReplyTo} ->
            send(ReplyTo, {reply, Id, Count}),
            server(Count, Notes);
        {notes, ReplyTo} when is_tuple(ReplyTo), is_pid(element(1, ReplyTo)) ->
            send(ReplyTo, lists:reverse(Notes)),
            server(Count, Notes);
        stop -> ok
    end.

server_with(Setup) ->
    send(Setup, {self(), notes}),
    server(0, []).

take_oldest(0, Got) ->
    lists:reverse(Got);
take_oldest(K, Got) ->
    Text = receive
        {b, N} -> "b" ++ integer_to_list(N);
        {a, S} -> "a:" ++ binary_to_list(S)
    end,
    take_oldest(K - 1, [list_to_binary(Text) | Got]).

three_first() ->
    receive
        {c, 3} -> mercury_write_line({small, 3});
        {c, N} -> mercury_write_line({large, N})
    end.

any_first() ->
    receive
        {c, N} -> mercury_write_line({large, N});
        {c, 3} -> mercury_write_line({small, 3})
    end.

reply_to(Id) ->
    receive
        {replies, {reply, Id, Count}} ->
            io:format("reply to ~w: ~w~n", [Id, Count])
    end.

main(_) ->
    Self = self(),
    send({Self, b}, 1),
    send({Self, a}, <<"two">>),
    send({Self, b}, 3),
    send({Self, a}, <<"four">>),
    mercury_write_line(take_oldest(4, [])),
    [send({Self, c}, N) || N <- [3, 30, 3]],
    three_first(),
    three_first(),
    any_first(),
    send({Self, c}, 7),
    send({Self, c}, 8),
    receive {c, 8} -> io:format("selected 8~n") end,
    receive
        {c, M} -> io:format("left queued: ~w~n", [M])
    after 0 -> io:format("nothing left~n")
    end,
    S = spawn(fun() -> server_with({Self, setup}) end),
    Notes = receive {setup, Box} -> Box end,
    S ! {increment, 5},
    send(Notes, {note, <<"hello">>}),
    S ! {get, 1, {Self, replies}},
    S ! {increment, 10},
    S ! {get, 2, {Self, replies}},
    reply_to(2),
    reply_to(1),
    send(Notes, {note, <<"world">>}),
    S ! {notes, {Self, notes_back}},
    receive {notes_back, Texts} -> mercury_write_line(Texts) end,
    S ! stop,
    receive
        {replies, _} -> io:format("a stray reply~n")
    after 10 -> io:format("timeout~n")
    end.
