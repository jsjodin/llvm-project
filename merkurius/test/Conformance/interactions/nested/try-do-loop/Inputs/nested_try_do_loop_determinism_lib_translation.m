% The feature and inner context of run-determinism.m.
:- module nested_try_do_loop_determinism_lib_translation.
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


inner_run(N) = string.join_list(" | ",
    list.map((func(K) = use(make(K))), [N, N + 1])) ++ " | last " ++
    use(make(N + 20)).
