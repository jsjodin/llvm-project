% The feature and inner context of run-try-goals.m.
:- module nested_nondet_polymorphic_try_goals_lib.
:- interface.
:- import_module string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module exception, int, list.

make(N) = N.

use(N) = S :-
    promise_equivalent_solutions [S] (
        ( try []
            risky(N, R)
        then
            S = "ok " ++ string.int_to_string(R)
        catch Message ->
            S = "caught " ++ Message
        )
    ).

:- pred risky(int::in, int::out) is det.

risky(N, R) :-
    ( if N mod 2 = 0 then
        throw("even " ++ string.int_to_string(N))
    else
        R = N * 3
    ).

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
