% The translation of run-shared-and-local.m's loops: Offset is passed to
% both loop predicates; T, I and Y are each loop's own.
:- module shared_and_local_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    Offset = 100,
    first(1, 3, Offset, Ys),
    second(1, 2, Offset, Zs),
    io.write(Ys, !IO), io.nl(!IO),
    io.write(Zs, !IO), io.nl(!IO).

:- pred first(int::in, int::in, int::in, list(int)::out) is det.
first(I, Hi, Offset, Ys) :-
    ( if I > Hi then
        Ys = []
    else
        T = I * I, Y = T + Offset, Ys = [Y | Ys1], first(I + 1, Hi, Offset, Ys1)
    ).

:- pred second(int::in, int::in, int::in, list(int)::out) is det.
second(I, Hi, Offset, Ys) :-
    ( if I > Hi then
        Ys = []
    else
        T = I - Offset, Ys = [T | Ys1], second(I + 1, Hi, Offset, Ys1)
    ).
