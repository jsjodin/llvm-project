% The feature and inner context of run-equivalence-types.m.
:- module nested_nondet_do_loop_equivalence_types_lib.
:- interface.
:- import_module list, pair, string.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).

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
