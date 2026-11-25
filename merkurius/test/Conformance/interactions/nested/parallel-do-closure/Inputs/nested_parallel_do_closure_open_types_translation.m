% The translation of run-open-types.m in closure in parallel-do for mmc: the
% same program in plain Mercury, which must print the same.
:- module nested_parallel_do_closure_open_types_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, nested_parallel_do_closure_open_types_lib_translation, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

:- func run(int) = string.

run(N) = string.join_list(" | ",
    list.map((func(I) = use_nested(make_nested(N + I))), 0 .. 3)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
