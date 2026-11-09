% Factorials from 0! to 20!, the largest that fits in a 64-bit int. fact/1
% is a function whose value is an if-then-else's; table/3 loops by a tail
% call, which runs in constant stack space.

:- module factorial.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func fact(int) = int.
fact(N) = ( if N =< 0 then 1 else N * fact(N - 1) ).

:- pred table(int::in, int::in, io::di, io::uo) is det.
table(I, N, !IO) :-
    ( if I > N then
        true
    else
        io.write_int(I, !IO),
        io.write_string("! = ", !IO),
        io.write_int(fact(I), !IO),
        io.nl(!IO),
        table(I + 1, N, !IO)
    ).

main(!IO) :-
    table(0, 20, !IO).
