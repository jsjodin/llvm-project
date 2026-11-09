% The first 21 Fibonacci numbers: a doubly recursive function, and a loop
% whose then part ends with a tail call.

:- module fib.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func fib(int) = int.
fib(N) = ( if N < 2 then N else fib(N - 1) + fib(N - 2) ).

:- pred loop(int::in, io::di, io::uo) is det.
loop(I, !IO) :-
    ( if I =< 20 then
        io.write_string("fib(", !IO),
        io.write_int(I, !IO),
        io.write_string(") = ", !IO),
        io.write_int(fib(I), !IO),
        io.nl(!IO),
        loop(I + 1, !IO)
    else
        true
    ).

main(!IO) :-
    loop(0, !IO).
