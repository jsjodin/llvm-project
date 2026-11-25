% The feature and inner context of run-try-goals.m.
:- module nested_closure_typeclass_method_try_goals_lib.
:- interface.
:- import_module string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module exception, int.

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
