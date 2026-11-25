% The translation of run-open-types.m in try for mmc: the same
% program in plain Mercury, which must print the same.
:- module try_open_types_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, string, univ.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = (1 .. N) ++ [N * 10].

use(L) =
    ( if L = [A, B | Rest] then
        string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        "short"
    ).

    % A value thrown as an exception, caught and used.
:- pred thrower(int::in, string::out) is det.

thrower(N, S) :-
    V = make(N),
    ( if N mod 2 = 0 then
        throw(V)
    else
        S = "returned " ++ use(V)
    ).

:- func run(int) = string.

run(N) = S :-
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

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
