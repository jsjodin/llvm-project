% The feature and inner context of run-determinism.m.
:- module nested_do_loop_parallel_do_determinism_lib.
:- interface.
:- import_module list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, int, solutions.

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

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make(N + I),
        !A ^ unsafe_elem(I) := use(V)
    ).


inner_run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).
