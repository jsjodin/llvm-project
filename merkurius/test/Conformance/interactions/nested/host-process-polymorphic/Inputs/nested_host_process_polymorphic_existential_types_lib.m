% The feature and inner context of run-existential-types.m.
:- module nested_host_process_polymorphic_existential_types_lib.
:- interface.
:- import_module string.

:- typeclass showable(T) where [
    func show(T) = string
].

:- instance showable(int).
:- instance showable(string).

    % A value of a hidden type, with its dictionary.
:- type fdata
    --->    some [T] boxed(T) => showable(T).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, list.

:- instance showable(int) where [
    show(I) = "int " ++ string.int_to_string(I)
].
:- instance showable(string) where [
    show(S) = "string " ++ S
].

make(N) = D :-
    ( if N mod 2 = 0 then
        D = 'new boxed'(N * 7)
    else
        D = 'new boxed'(string.duplicate_char('z', N))
    ).

use(boxed(X)) = show(X).

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
