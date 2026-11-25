% The feature and inner context of run-user-defined-equality.m.
:- module nested_nondet_try_user_defined_equality_lib.
:- interface.
:- import_module list, string.

    % A set as a list in any order, equal by its members.
:- type fdata
    --->    set(list(int))
    where equality is same_members.

:- pred same_members(fdata::in, fdata::in) is semidet.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module exception, int, univ.

same_members(A, B) :-
    promise_equivalent_solutions [SA, SB] (
        A = set(LA),
        B = set(LB),
        list.sort_and_remove_dups(LA, SA),
        list.sort_and_remove_dups(LB, SB)
    ),
    SA = SB.

make(N) = set(list.reverse(1 .. N) ++ [1]).

use(D) = S :-
    % Built first, so that the unification compares two values, by the
    % type's equality.
    Expected = set([3, 2, 1]),
    ( if D = Expected then
        Equal = "equal to [1, 2, 3]"
    else
        Equal = "not equal to [1, 2, 3]"
    ),
    promise_equivalent_solutions [Count] (
        D = set(L),
        list.sort_and_remove_dups(L, Members),
        Count = list.length(Members)
    ),
    S = string.int_to_string(Count) ++ " members, " ++ Equal.

    % A value thrown as an exception, caught and used.
:- pred thrower(int::in, string::out) is det.

thrower(N, S) :-
    V = make(N),
    ( if N mod 2 = 0 then
        throw(V)
    else
        S = "returned " ++ use(V)
    ).


inner_run(N) = S :-
    promise_equivalent_solutions [R] (
        try(thrower(N), R)
    ),
    (
        R = succeeded(S)
    ;
        R = exception(E),
        ( if univ_to_type(E, V) then
            S = "caught " ++ use(V)
        else
            S = "caught something else"
        )
    ).
