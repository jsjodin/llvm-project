% The feature and inner context of run-determinism.m.
:- module nested_message_typeclass_method_determinism_lib.
:- interface.
:- import_module list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, solutions.

make(N) = 1 .. N + 2.

use(L) = S :-
    solutions(even_member(L), Evens),
    ( if first_over(L, 2, F) then
        First = string.int_to_string(F)
    else
        First = "none"
    ),
    S = "evens " ++ string.join_list(",",
        list.map(string.int_to_string, Evens)) ++ ", first over 2 " ++ First.

:- pred even_member(list(int)::in, int::out) is nondet.

even_member(L, X) :-
    list.member(X, L),
    X mod 2 = 0.

:- pred first_over(list(int)::in, int::in, int::out) is semidet.

first_over([X | Xs], Limit, F) :-
    ( if X > Limit then
        F = X
    else
        first_over(Xs, Limit, F)
    ).

:- typeclass runner(T) where [
    func produce(T, int) = fdata,
    func consume(T, fdata) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make(N),
    consume(plain, V) = use(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make(N + K),
    consume(offset(_), V) = "offset " ++ use(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).


inner_run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).
