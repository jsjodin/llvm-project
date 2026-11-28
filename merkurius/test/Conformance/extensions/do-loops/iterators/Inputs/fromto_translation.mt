% The translation of run-fromto.mt's loops. Without a for, a loop is the
% disjunction ( Ends ; Steps, Body, Recurse ), a switch on the fromto that
% ends it.
:- module fromto_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list.

:- type go ---> continue ; stop.

main(!IO) :-
    sum(1, 4, 0.0@S@, Sum),
    io.write_@T@(Sum, !IO), io.nl(!IO),
    reversed([3, 1, 2], [], Reversed),
    io.write(Reversed, !IO), io.nl(!IO),
    power(continue, 1, Power),
    io.write_int(Power, !IO), io.nl(!IO).

:- pred sum(int::in, int::in, @T@::in, @T@::out) is det.
sum(I, Hi, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        sum(I + 1, Hi, S0 + @FROM_INT@(I) / 2.0@S@, Sum)
    ).

:- pred reversed(list(int)::in, list(int)::in, list(int)::out) is det.
reversed([], R, R).
reversed([X | Xs], R0, R) :- reversed(Xs, [X | R0], R).

:- pred power(go::in, int::in, int::out) is det.
power(stop, P, P).
power(continue, P0, P) :-
    P1 = P0 * 3,
    C = ( if P1 > 100 then stop else continue ),
    power(C, P1, P).
