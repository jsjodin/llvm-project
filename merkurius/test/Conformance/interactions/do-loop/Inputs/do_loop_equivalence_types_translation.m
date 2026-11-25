% The translation of run-equivalence-types.m in do-loop for mmc: the same
% program in plain Mercury, which must print the same.
:- module do_loop_equivalence_types_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair, string.

:- type name == string.
:- type fdata == pair(int, list(name)).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = N - list.map((func(I) = "n" ++ string.int_to_string(I)),
    1 .. N).

use(K - Names) = string.int_to_string(K) ++ ": " ++
    string.join_list(" ", Names).

:- func run(int) = string.

run(N) = string.join_list(" | ",
    list.map((func(K) = use(make(K))), [N, N + 1])) ++ " | last " ++
    use(make(N + 20)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
