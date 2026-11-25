% The translation of run-flat-values.m in gpu-process for mmc: the same
% program in plain Mercury, which must print the same.
:- module gpu_process_flat_values_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, int8, list, string, uint16.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type wrapped
    --->    wrapped(int).

    % A record of one functor of numbers, an enumeration and a notag.
:- type fdata
    --->    fdata(colour, wrapped, int8, uint16, float).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = fdata(
    ( if N mod 3 = 0 then red else if N mod 3 = 1 then green else blue ),
    wrapped(N * 100), int8.det_from_int(-N), uint16.det_from_int(N * 1000),
    float(N) * 0.25).

use(fdata(C, wrapped(W), I, U, F)) =
    colour_code(C) + W + int8.to_int(I) + uint16.to_int(U) +
    truncate_to_int(F * 100.0).

:- func colour_code(colour) = int.

colour_code(red) = 1.
colour_code(green) = 2.
colour_code(blue) = 3.

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
