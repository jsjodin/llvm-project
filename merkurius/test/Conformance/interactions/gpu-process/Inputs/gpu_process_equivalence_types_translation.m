% The translation of run-equivalence-types.m in gpu-process for mmc: the same
% feature in plain Mercury, which must print the same.
:- module gpu_process_equivalence_types_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type point
    --->    point(int, int).
:- type distance == int.
:- type fdata == point.

:- func make(int) = fdata.
:- func use(fdata) = distance.

make(N) = point(N, N * 2).

use(point(X, Y)) = X * X + Y * Y.

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
