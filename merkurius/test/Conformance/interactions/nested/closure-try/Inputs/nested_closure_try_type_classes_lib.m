% The feature and inner context of run-type-classes.m.
:- module nested_closure_try_type_classes_lib.
:- interface.
:- import_module string.

:- typeclass shape(T) where [
    func area(T) = int,
    func name(T) = string
].

:- type fdata
    --->    rect(int, int)
    ;       square(int).

:- instance shape(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module exception, int, list, univ.

:- instance shape(fdata) where [
    area(rect(W, H)) = W * H,
    area(square(S)) = S * S,
    name(rect(_, _)) = "rect",
    name(square(_)) = "square"
].

make(N) = ( if N mod 2 = 0 then square(N) else rect(N, N + 1) ).

use(D) = describe([D, square(1)]).

    % Through a constraint, so that the method is called by its dictionary.
:- func describe(list(T)) = string <= shape(T).
:- pragma no_inline(describe/1).

describe(Shapes) = string.join_list(", ",
    list.map((func(S) = name(S) ++ "=" ++ string.int_to_string(area(S))),
        Shapes)).

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
