% The feature and inner context of run-closures-as-data.m.
:- module nested_typeclass_method_polymorphic_closures_as_data_lib.
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
