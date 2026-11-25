% The feature and inner context of run-closures-as-data.m.
:- module nested_host_process_try_closures_as_data_lib.
:- interface.
:- import_module string.

    % Functions as data, which capture the argument of make.
:- type fdata
    --->    fns(func(int) = int, func(int) = string).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module exception, int, univ.

make(N) = fns((func(X) = X * N + 1),
    (func(X) = string.int_to_string(X) ++ "/" ++ string.int_to_string(N))).

use(fns(F, G)) = string.int_to_string(apply(F, 10)) ++ " " ++ G(F(2)).

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
