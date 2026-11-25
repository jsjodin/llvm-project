% The translation of run-teams-loop.m's loops, without their directives.
:- module teams_loop_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int.

:- pred stats_loop(int::in, int::in, int::in, int::out, int::in, int::out,
    int::in, int::out) is det.
stats_loop(I, N, S0, Sum, L0, Lo, H0, Hi) :-
    ( if I > N then
        Sum = S0, Lo = L0, Hi = H0
    else
        V = (I * 37) mod 101 - 50,
        stats_loop(I + 1, N, S0 + V, Sum, int.min(L0, V), Lo,
            int.max(H0, V), Hi)
    ).

:- pred sum_loop(int::in, int::in, array(float)::array_ui, float::in,
    float::out) is det.
sum_loop(I, Hi, A, S0, Sum) :-
    ( if I > Hi then
        Sum = S0
    else
        sum_loop(I + 1, Hi, A, S0 + A ^ elem(I), Sum)
    ).

:- pred divide_loop(int::in, int::in, float::in, array(float)::array_di,
    array(float)::array_uo) is det.
divide_loop(I, Hi, Sum, !A) :-
    ( if I > Hi then
        true
    else
        !A ^ elem(I) := !.A ^ elem(I) / Sum,
        divide_loop(I + 1, Hi, Sum, !A)
    ).

main(!IO) :-
    stats_loop(1, 100000, 0, Sum, max_int, Lo, min_int, Hi),
    io.write({Sum, Lo, Hi}, !IO), io.nl(!IO),
    array.init(4096, 0.5, A0),
    sum_loop(0, 4095, A0, 0.0, S),
    divide_loop(0, 4095, S, A0, A),
    io.write_float(A ^ elem(0) * 4096.0, !IO), io.nl(!IO),
    io.write_float(A ^ elem(4095) * 4096.0, !IO), io.nl(!IO).
