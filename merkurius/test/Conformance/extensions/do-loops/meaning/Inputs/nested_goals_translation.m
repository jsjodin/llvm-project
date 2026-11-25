% The translation of run-nested-goals.m for mmc: each loop a predicate.
:- module nested_goals_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred sum_loop(list(int)::in, int::in, int::out) is det.

sum_loop([], S, S).
sum_loop([X | Xs], S0, S) :-
    sum_loop(Xs, S0 + X, S).

:- pred all_odd(list(int)::in) is semidet.

all_odd([]).
all_odd([Y | Ys]) :-
    Y mod 2 = 1,
    all_odd(Ys).

main(!IO) :-
    Sum = (func(Xs) = S :- sum_loop(Xs, 0, S)),
    io.write_int(Sum([1, 2, 3]), !IO),
    io.nl(!IO),
    L = [1, 3, 5],
    ( if not all_odd(L) then
        io.write_string("not all odd\n", !IO)
    else
        io.write_string("all odd\n", !IO)
    ),
    ( if \+ all_odd([2 | L]) then
        io.write_string("not all odd\n", !IO)
    else
        io.write_string("all odd\n", !IO)
    ).
