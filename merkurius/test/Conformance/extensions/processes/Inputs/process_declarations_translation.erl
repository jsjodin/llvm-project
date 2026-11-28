%% The translation of run-process-declarations.mt into Erlang: each spawn,
%% by kind or by a closure in a variable, is spawn of the kind's function.
-module(process_declarations_translation).
-export([main/1]).
-mode(compile).
% A template, as the test is: FLOAT_TYPE is the instance's type.
-define(FLOAT_TYPE, @T@).
-include("../../gpu-processes/Inputs/mercury_write.hrl").

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
