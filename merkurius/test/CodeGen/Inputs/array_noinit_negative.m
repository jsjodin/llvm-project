% array.unsafe_noinit of a negative size throws (array-noinit-errors.m).

:- module array_noinit_negative.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, int, list, string.

main(!IO) :-
    io.command_line_arguments(Args, !IO),
    N = -1 - list_length(Args),
    array.unsafe_noinit(N, A : array(int)),
    io.write_int(array.size(A), !IO),
    io.nl(!IO).

:- func list_length(list(string)) = int.
list_length([]) = 0.
list_length([_ | T]) = 1 + list_length(T).
