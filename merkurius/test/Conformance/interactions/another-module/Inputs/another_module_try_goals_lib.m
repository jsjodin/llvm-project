% The feature of run-try-goals.m, as another module.
:- module another_module_try_goals_lib.
:- interface.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module exception, int, string.

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
