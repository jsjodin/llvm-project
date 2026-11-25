% The translation of run-open-types.m in parallel-do for mmc: the same
% program in plain Mercury, which must print the same.
:- module parallel_do_open_types_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = (1 .. N) ++ [N * 10].

use(L) =
    ( if L = [A, B | Rest] then
        string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        "short"
    ).

:- func run(int) = string.

run(N) = string.join_list(" | ",
    list.map((func(I) = use(make(N + I))), 0 .. 3)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
