% The feature and inner context of run-user-defined-equality.m.
:- module nested_parallel_do_typeclass_method_user_defined_equality_lib.
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
