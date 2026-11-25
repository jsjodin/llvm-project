% The translation of run-foreach.m's loops. The pattern K - V is a
% unification at the start of the body.
:- module foreach_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair, string.

main(!IO) :-
    Words = ["one", "two", "three"],
    lengths(Words, Lengths),
    io.write(Lengths, !IO), io.nl(!IO),
    writes(Words, !IO),
    joined([1 - "a", 2 - "b"], Joined),
    io.write(Joined, !IO), io.nl(!IO).

:- pred lengths(list(string)::in, list(int)::out) is det.
lengths([], []).
lengths([W | Ws], [L | Ls]) :- L = string.length(W), lengths(Ws, Ls).

:- pred writes(list(string)::in, io::di, io::uo) is det.
writes([], !IO).
writes([W | Ws], !IO) :- io.write_string(W, !IO), io.nl(!IO), writes(Ws, !IO).

:- pred joined(list(pair(int, string))::in, list(string)::out) is det.
joined([], []).
joined([P | Ps], [S | Ss]) :-
    P = K - V, S = string.int_to_string(K) ++ V, joined(Ps, Ss).
