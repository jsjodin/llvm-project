% The feature and inner context of run-try-goals.m.
:- module nested_do_loop_parallel_do_try_goals_lib.
:- interface.
:- import_module string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, exception, int, list.

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
