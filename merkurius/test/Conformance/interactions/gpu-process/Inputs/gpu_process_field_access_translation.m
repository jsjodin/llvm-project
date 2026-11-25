% The translation of run-field-access.m in gpu-process for mmc: the same
% program in plain Mercury, which must print the same.
:- module gpu_process_field_access_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type fdata
    --->    account(
                id :: int,
                balance :: int,
                limit :: int
            ).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = account(N, N * 10, 50).

use(A0) = A ^ id * 10000 + A ^ balance * 10 + A ^ limit :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ limit := A1 ^ limit - A0 ^ id.

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
