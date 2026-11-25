%% Writes Erlang terms as Merkurius' io.write writes the Mercury terms they
%% stand for, for the Erlang translations of the processes tests: a number
%% as itself, an atom as a constant, a tuple {f, A, B} as f(A, B),
%% {tuple, A, B} as {A, B}, a list as [A, B], a binary as a string in
%% double quotes, a pair {'-', A, B} as A - B, and {suffixed, N, "i32"} as
%% a sized integer, 5i32.

mercury_write_line(T) ->
    mercury_write(T),
    io:format("~n").

mercury_write(T) ->
    io:format("~s", [mercury_string(T)]).

mercury_string(T) when is_integer(T) ->
    integer_to_list(T);
mercury_string(T) when is_float(T) ->
    io_lib:format("~w", [T]);
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
