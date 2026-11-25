% The translation of run-lockstep.m's loops: one predicate per loop, the
% inner loop's called from the outer one's body. Lists of different
% lengths are a software error.
:- module lockstep_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, require.

main(!IO) :-
    zs(0, 2, [10, 20, 30], [1, 2, 3], Zs),
    io.write(Zs, !IO), io.nl(!IO),
    rows(1, 3, Table),
    io.write(Table, !IO), io.nl(!IO).

:- pred zs(int::in, int::in, list(int)::in, list(int)::in, list(int)::out)
    is det.
zs(I, Hi, Xs, Ys, Zs) :-
    ( if I > Hi then
        ( if Xs = [], Ys = [] then Zs = [] else unexpected($pred, "lengths") )
    else
        ( if Xs = [X | Xs1], Ys = [Y | Ys1] then
            Zs = [I + X * Y | Zs1],
            zs(I + 1, Hi, Xs1, Ys1, Zs1)
        else
            unexpected($pred, "lengths")
        )
    ).

:- pred rows(int::in, int::in, list(list(int))::out) is det.
rows(I, Hi, Rows) :-
    ( if I > Hi then
        Rows = []
    else
        row(I, 1, 3, Row), Rows = [Row | Rows1], rows(I + 1, Hi, Rows1)
    ).

:- pred row(int::in, int::in, int::in, list(int)::out) is det.
row(I, J, Hi, Row) :-
    ( if J > Hi then Row = [] else Row = [I * J | Row1], row(I, J + 1, Hi, Row1) ).
