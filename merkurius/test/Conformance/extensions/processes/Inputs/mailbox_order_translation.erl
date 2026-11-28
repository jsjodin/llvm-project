%% The translation of run-mailbox-order.m into Erlang. The process's two
%% mailboxes are its one Erlang mailbox, each message tagged with its
%% mailbox, a or b, so that a selective receive over both takes the
%% oldest message that a clause matches, as the test's selector does.
-module(mailbox_order_translation).
-export([main/1]).
-mode(compile).
-include("../../gpu-processes/Inputs/mercury_write.hrl").

drain(Tag) ->
    drain(Tag, []).

drain(Tag, Got) ->
    receive
        {Tag, M} -> drain(Tag, [M | Got])
    after 0 -> lists:reverse(Got)
    end.

show(Label, R) ->
    io:format("~s: ", [Label]),
    mercury_write_line(R).

send_all(Tag, Ms) ->
    [self() ! {Tag, M} || M <- Ms].

urgent() ->
    receive {a, {urgent, N}} -> {message, N} after 0 -> timeout end.

receive_all(Events) ->
    receive
        {a, M} -> receive_all([{from_a, M} | Events]);
        {b, N} when N rem 2 =:= 0 -> receive_all([{from_b, N} | Events])
    after 0 -> lists:reverse(Events)
    end.

main(_) ->
    send_all(a, [{data, 1}, {data, 2}, {data, 3}]),
    mercury_write_line(drain(a)),
    R0 = receive {a, M0} -> {message, M0} after 0 -> timeout end,
    show("empty", R0),
    send_all(a, [{data, 4}, {urgent, 10}, {data, 5}, {urgent, 11}, done]),
    show("urgent", urgent()),
    show("urgent", urgent()),
    show("urgent", urgent()),
    mercury_write_line(drain(a)),
    self() ! {b, 1},
    self() ! {a, {data, 6}},
    self() ! {b, 2},
    self() ! {a, {data, 7}},
    self() ! {b, 4},
    mercury_write_line(receive_all([])),
    R4 = receive {b, M4} -> {message, M4} after 0 -> timeout end,
    show("left in b", R4).
