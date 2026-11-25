% The feature and inner context of run-field-access.m.
:- module nested_try_do_loop_field_access_lib.
:- interface.
:- import_module list, string.

:- type fdata
    --->    account(
                owner :: string,
                balance :: int,
                history :: list(int)
            ).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int.

make(N) = account("owner" ++ string.int_to_string(N), N * 10, []).

use(A0) = S :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ history := [A0 ^ balance | A1 ^ history],
    S = A ^ owner ++ " " ++ string.int_to_string(A ^ balance) ++ " " ++
        string.join_list(",", list.map(string.int_to_string, A ^ history)).

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
