%% The translation of nyi-several-mailboxes.m into Erlang: the kind's two
%% mailboxes are its one Erlang mailbox, each message tagged with its
%% mailbox, control or data.
-module(several_mailboxes_translation).
-export([main/1]).
-mode(compile).
-include("../../gpu-processes/Inputs/mercury_write.hrl").

summer(Sum) ->
    receive
        {control, {stop, ReplyTo}} -> ReplyTo ! {back, Sum};
        {data, {data, N}} -> summer(Sum + N)
    end.

main(_) ->
    S = spawn(fun() -> summer(0) end),
    S ! {data, {data, 1}},
    S ! {data, {data, 2}},
    S ! {control, {stop, self()}},
    R = receive {back, Sum} -> {message, Sum} end,
    mercury_write_line(R).
