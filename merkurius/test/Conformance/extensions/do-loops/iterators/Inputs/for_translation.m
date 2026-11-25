% The translation of run-for.m's loops, written by hand as their meaning:
% ( if I > Hi then Ends else Steps, Body, Recurse ), with < for a negative
% step. The loop to max_int counts down a trip count, as merkc's does.
:- module for_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    up(1, 5, Up), io.write(Up, !IO), io.nl(!IO),
    strided(1, 10, Strided), io.write(Strided, !IO), io.nl(!IO),
    down(10, 1, Down), io.write(Down, !IO), io.nl(!IO),
    N = 3,
    up_id(N, N - 1, None), io.write(None, !IO), io.nl(!IO),
    count(max_int - (max_int - 2) + 1, 0, C),
    io.write_int(C, !IO), io.nl(!IO).

:- pred up(int::in, int::in, list(int)::out) is det.
up(I, Hi, Xs) :-
    ( if I > Hi then Xs = [] else Xs = [I * I | T], up(I + 1, Hi, T) ).

:- pred strided(int::in, int::in, list(int)::out) is det.
strided(I, Hi, Xs) :-
    ( if I > Hi then Xs = [] else Xs = [I | T], strided(I + 3, Hi, T) ).

:- pred down(int::in, int::in, list(int)::out) is det.
down(I, Hi, Xs) :-
    ( if I < Hi then Xs = [] else Xs = [I | T], down(I - 4, Hi, T) ).

:- pred up_id(int::in, int::in, list(int)::out) is det.
up_id(I, Hi, Xs) :-
    ( if I > Hi then Xs = [] else Xs = [I | T], up_id(I + 1, Hi, T) ).

:- pred count(int::in, int::in, int::out) is det.
count(Trips, C0, C) :-
    ( if Trips =< 0 then C = C0 else count(Trips - 1, C0 + 1, C) ).
