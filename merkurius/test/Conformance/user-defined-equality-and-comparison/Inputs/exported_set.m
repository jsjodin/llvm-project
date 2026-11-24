:- module exported_set.
:- interface.
:- import_module list.

:- type set(T)
    --->    set(list(T))
            where equality is set_equals.

:- pred set_equals(set(T)::in, set(T)::in) is semidet.

:- implementation.

set_equals(S1, S2) :-
    promise_equivalent_solutions [L1, L2] (
        S1 = set(L1),
        S2 = set(L2)
    ),
    all_in(L1, L2),
    all_in(L2, L1).

:- pred all_in(list(T)::in, list(T)::in) is semidet.
all_in([], _).
all_in([X | Xs], L) :-
    list.member(X, L),
    all_in(Xs, L).
