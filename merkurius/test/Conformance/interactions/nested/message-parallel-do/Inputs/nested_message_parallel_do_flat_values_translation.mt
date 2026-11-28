% The translation of run-flat-values.mt in parallel-do in message for mmc: the
% same program in plain Mercury, which must print the same.
:- module nested_message_parallel_do_flat_values_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_message_parallel_do_flat_values_lib_translation, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, use_nested(make_nested(N)) ++ " | " ++ use_nested(make_nested(N + 1)), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
