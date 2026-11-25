% The feature and inner context of run-field-access.m.
:- module nested_typeclass_method_polymorphic_field_access_lib.
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

    % Polymorphic code that carries the values, in lists, pairs and maybes,
    % compiled once for every type.
:- pred keep(T::in, T::out) is det.
:- pragma no_inline(keep/2).

keep(X, X).

:- type holder(T)
    --->    holder(T, int)
    ;       nothing.

:- func second(list(T), T) = T.
:- pragma no_inline(second/2).

second(Xs, Default) =
    ( if Xs = [_, X | _] then X else Default ).


inner_run(N) = S :-
    keep([make(N), make(N + 1)], Vs),
    keep(holder(make(N + 2), N), H),
    ( if H = holder(V, _) then
        Held = use(V)
    else
        Held = "nothing"
    ),
    S = use(second(list.reverse(Vs), make(0))) ++ " | " ++ Held.
