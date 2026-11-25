% The feature and inner context of run-equivalence-types.m.
:- module nested_try_polymorphic_equivalence_types_lib.
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
