% The feature and inner context of run-user-defined-equality.m.
:- module nested_polymorphic_do_loop_user_defined_equality_lib.
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
:- import_module int.

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

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.

inner_run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).
