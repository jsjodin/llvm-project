%% Writes Erlang terms as Merkurius' io.write writes the Mercury terms they
%% stand for, for the Erlang translations of the processes tests: a number
%% as itself, an atom as a constant, a tuple {f, A, B} as f(A, B),
%% {tuple, A, B} as {A, B}, a list as [A, B], a binary as a string in
%% double quotes, a pair {'-', A, B} as A - B, and {suffixed, N, "i32"} as
%% a sized integer, 5i32. A float is written as a float of the type that
%% FLOAT_TYPE names, float unless the translation defines it before it
%% includes this, as a template's translation does, -define(FLOAT_TYPE,
%% @T@): a float32 as the oracle's stand-in writes it, float32_value(X),
%% which oracle.py writes as merkc's programs do. ?R(X) rounds X to the
%% type, for each operation of a float32 translation.

-ifndef(FLOAT_TYPE).
-define(FLOAT_TYPE, float).
-endif.
-define(R(X), round_float(?FLOAT_TYPE, X)).
-compile({nowarn_unused_function, [round_float/2]}).

mercury_write_line(T) ->
    mercury_write(T),
    io:format("~n").

mercury_write(T) ->
    io:format("~s", [mercury_string(T)]).

mercury_string(T) when is_integer(T) ->
    integer_to_list(T);
mercury_string(T) when is_float(T) ->
    float_string(?FLOAT_TYPE, T);
mercury_string(T) when is_atom(T) ->
    atom_to_list(T);
mercury_string(<<>>) ->
    "\"\"";
mercury_string(T) when is_binary(T) ->
    io_lib:format("~p", [binary_to_list(T)]);
mercury_string(L) when is_list(L) ->
    ["[", mercury_arguments(L), "]"];
mercury_string({suffixed, N, Suffix}) ->
    [integer_to_list(N), Suffix];
mercury_string({'-', A, B}) ->
    [mercury_string(A), " - ", mercury_string(B)];
mercury_string(T) when is_tuple(T), element(1, T) =:= tuple ->
    ["{", mercury_arguments(tl(tuple_to_list(T))), "}"];
mercury_string(T) when is_tuple(T) ->
    [F | Args] = tuple_to_list(T),
    [atom_to_list(F), "(", mercury_arguments(Args), ")"].

mercury_arguments(Args) ->
    lists:join(", ", [mercury_string(A) || A <- Args]).

float_string(float, T) ->
    io_lib:format("~w", [T]);
float_string(float32, T) ->
    io_lib:format("float32_value(~w)", [T]).

round_float(float, X) ->
    X;
round_float(float32, X) ->
    <<Y:32/float>> = <<X:32/float>>,
    Y.
