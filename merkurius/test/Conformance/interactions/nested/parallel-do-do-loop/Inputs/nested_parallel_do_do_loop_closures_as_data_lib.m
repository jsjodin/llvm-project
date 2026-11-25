% The feature and inner context of run-closures-as-data.m.
:- module nested_parallel_do_do_loop_closures_as_data_lib.
:- interface.
:- import_module string.

    % Functions as data, which capture the argument of make.
:- type fdata
    --->    fns(func(int) = int, func(int) = string).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, list.

make(N) = fns((func(X) = X * N + 1),
    (func(X) = string.int_to_string(X) ++ "/" ++ string.int_to_string(N))).

use(fns(F, G)) = string.int_to_string(apply(F, 10)) ++ " " ++ G(F(2)).

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.

inner_run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).
