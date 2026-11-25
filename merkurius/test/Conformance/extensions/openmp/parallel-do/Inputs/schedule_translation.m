% The translation of run-schedule.m's loops, without their directives:
% each loop sums work(I) and counts its iterations.
:- module schedule_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func work(int) = int.
work(I) = steps(I mod 97, I).

:- func steps(int, int) = int.
steps(K, W) = ( if K =< 0 then W else steps(K - 1, (W * 31 + 7) mod 1000003) ).

:- pred sum(int::in, int::in, int::in, int::out, int::in, int::out) is det.
sum(I, Hi, S0, S, C0, C) :-
    ( if I > Hi then
        S = S0, C = C0
    else
        sum(I + 1, Hi, S0 + work(I), S, C0 + 1, C)
    ).

main(!IO) :-
    sum(0, 999, 0, S1, 0, C1), io.write({S1, C1}, !IO), io.nl(!IO),
    sum(0, 999, 0, S2, 0, C2), io.write({S2, C2}, !IO), io.nl(!IO),
    sum(0, 999, 0, S3, 0, C3), io.write({S3, C3}, !IO), io.nl(!IO),
    sum(0, 999, 0, S4, 0, C4), io.write({S4, C4}, !IO), io.nl(!IO).
