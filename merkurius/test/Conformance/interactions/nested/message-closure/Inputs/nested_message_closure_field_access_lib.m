% The feature and inner context of run-field-access.m.
:- module nested_message_closure_field_access_lib.
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

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.

inner_run(N) = S :-
    Make = (func(K) = make(K)),
    V = apply(Make, N),
    Use = ((func) = use(V)),
    Both = list.map((func(K) = use(make(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).
