% Allocates in a target region without end, for omp-target-arena-limit-gpu.test.
:- module arena_forever.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- func forever(int, list(int)) = int.
forever(N, Xs) = ( if N < 0 then N else forever(N + 1, [N | Xs]) ).

main(!IO) :-
    %$omp target
    X = forever(0, []),
    io.write_int(X, !IO), io.nl(!IO).
