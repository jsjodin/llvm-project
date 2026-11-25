% The translation of run-nondet-body.m's loop.
:- module nondet_body_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred choices(list(int)::out) is multi.
choices(L) :- loop([1, 2], L).

:- pred loop(list(int)::in, list(int)::out) is multi.
loop([], []).
loop([X | Xs], [Y | Ys]) :- ( Y = X ; Y = X * 10 ), loop(Xs, Ys).

main(!IO) :-
    solutions(choices, Ls),
    io.write(Ls, !IO), io.nl(!IO).
