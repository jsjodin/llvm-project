% The translation of run-field-access.m in parallel-do in do-loop for mmc: the
% same program in plain Mercury, which must print the same.
:- module nested_do_loop_parallel_do_field_access_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_parallel_do_field_access_lib_translation, string.

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
    list.map((func(K) = use_nested(make_nested(K))), [N, N + 1])) ++ " | last " ++
    use_nested(make_nested(N + 20)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
