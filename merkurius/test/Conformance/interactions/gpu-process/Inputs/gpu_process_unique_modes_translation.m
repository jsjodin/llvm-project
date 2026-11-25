% The translation of run-unique-modes.m in gpu-process for mmc: the same
% program in plain Mercury, which must print the same.
:- module gpu_process_unique_modes_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = N + 3.

use(Size) = sum(0, squares(0, A0), 0) :-
    array.init(Size, 0, A0).

    % Updates the array in place, as its modes allow.
:- func squares(int, array(int)) = array(int).
:- mode squares(in, array_di) = array_uo is det.

squares(I, A0) = A :-
    ( if I < array.size(A0) then
        array.set(I, I * I, A0, A1),
        A = squares(I + 1, A1)
    else
        A = A0
    ).

:- func sum(int, array(int), int) = int.

sum(I, A, S) =
    ( if I < array.size(A) then sum(I + 1, A, S + A ^ elem(I)) else S ).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
