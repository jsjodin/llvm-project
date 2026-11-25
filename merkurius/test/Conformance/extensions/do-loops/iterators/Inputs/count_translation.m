% The translation of run-count.m's loops: Ends unifies the counter with
% Hi, Steps advances it.
:- module count_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    numbered([la, lb, lc], 1, N, Numbered),
    io.write(Numbered, !IO), io.nl(!IO),
    io.write_int(N, !IO), io.nl(!IO),
    counted([] : list(int), 10, M),
    io.write_int(M, !IO), io.nl(!IO),
    ( if counted([xa, ya], 0, 1) then
        io.write_string("ends at 1\n", !IO)
    else
        io.write_string("does not end at 1\n", !IO)
    ).

:- type letter ---> la ; lb ; lc.
:- type xy ---> xa ; ya.

:- pred numbered(list(letter)::in, int::in, int::out,
    list({int, letter})::out) is det.
numbered([], I, I - 1, []).
numbered([X | Xs], I, N, [{I, X} | Ps]) :- numbered(Xs, I + 1, N, Ps).

:- pred counted(list(T)::in, int::in, int::out) is det.
counted([], I, I - 1).
counted([_ | Xs], I, N) :- counted(Xs, I + 1, N).
