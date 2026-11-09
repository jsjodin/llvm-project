% Count to N. The recursive call is the procedure's last goal, so it is a
% musttail call: counting to a million needs no more stack than counting
% to ten.

:- module count.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred count(int::in, int::in, io::di, io::uo) is det.
count(I, N, !IO) :-
    ( if I > N then
        true
    else
        io.write_int(I, !IO),
        io.nl(!IO),
        count(I + 1, N, !IO)
    ).

main(!IO) :-
    count(1, 10, !IO).
