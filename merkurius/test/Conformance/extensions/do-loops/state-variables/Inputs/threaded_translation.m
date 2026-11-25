% The translation of run-threaded.m's loops: !Total is threaded through
% the loop predicate, and !.F, only read, is passed in.
:- module threaded_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    sums([3, 4, 5], 100, Total, 2, Scaled),
    io.write({Total, Scaled}, !IO), io.nl(!IO),
    writes([1, 2], !IO).

:- pred sums(list(int)::in, int::in, int::out, int::in, int::out) is det.
sums(Xs, !Total, Factor, Scaled) :-
    F = Factor,
    loop(Xs, F, !Total),
    Scaled = F * 10.

:- pred loop(list(int)::in, int::in, int::in, int::out) is det.
loop([], _, !Total).
loop([X | Xs], F, !Total) :- !:Total = !.Total + X * F, loop(Xs, F, !Total).

:- pred writes(list(int)::in, io::di, io::uo) is det.
writes([], !IO).
writes([X | Xs], !IO) :- io.write_int(X, !IO), io.nl(!IO), writes(Xs, !IO).
