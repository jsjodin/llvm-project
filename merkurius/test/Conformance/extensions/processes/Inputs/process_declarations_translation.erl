%% The translation of run-process-declarations.m into Erlang: each spawn,
%% by kind or by a closure in a variable, is spawn of the kind's function.
-module(process_declarations_translation).
-export([main/1]).
-mode(compile).
-include("mercury_write.hrl").

accumulator(Items) ->
    receive
        {add, X} -> accumulator([X | Items]);
        {get, ReplyTo} -> ReplyTo ! {items, lists:reverse(Items)}
    end.

ask(Box) ->
    Box ! {get, self()},
    receive {items, Items} -> mercury_write_line(Items) end.

main(_) ->
    Ints = spawn(fun() -> accumulator([]) end),
    Ints ! {add, 1},
    Ints ! {add, 2},
    ask(Ints),
    Strings = spawn(fun() -> accumulator([<<"start">>]) end),
    Strings ! {add, <<"more">>},
    ask(Strings),
    Kind = fun() -> accumulator([1.5]) end,
    Floats = spawn(Kind),
    Floats ! {add, 2.5},
    ask(Floats).
