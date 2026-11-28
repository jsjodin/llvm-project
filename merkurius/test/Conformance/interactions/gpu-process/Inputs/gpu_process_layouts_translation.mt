% The translation of run-layouts.mt in gpu-process for mmc: the same
% program in plain Mercury, which must print the same.
:- module gpu_process_layouts_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list, string.

    % More functors with arguments than a pointer has tags, and constants,
    % of fixed size: a flat union.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, int)
    ;       three(@T@, int)
    ;       four(int, int, int, int)
    ;       five(@T@)
    ;       six(int)
    ;       seven(int, @T@, int).

:- func make(int) = fdata.
:- func use(fdata) = int.

make(N) = D :-
    K = N mod 9,
    ( if K = 0 then D = c0
    else if K = 1 then D = one(N)
    else if K = 2 then D = two(N, -N)
    else if K = 3 then D = three(@FROM_INT@(N) / 2.0@S@, N)
    else if K = 4 then D = four(N, 1, 2, 3)
    else if K = 5 then D = five(@FROM_INT@(N) * 4.0@S@)
    else if K = 6 then D = six(N * N)
    else if K = 7 then D = seven(N, 0.5@S@, N)
    else D = c1
    ).

use(c0) = 1000.
use(c1) = 2000.
use(one(I)) = I.
use(two(I, J)) = I * 10 + J.
use(three(F, I)) = truncate_to_int(F * 10.0@S@) + I.
use(four(A, B, C, D)) = A * 1000 + B * 100 + C * 10 + D.
use(five(F)) = truncate_to_int(F).
use(six(I)) = I + 7.
use(seven(A, F, B)) = A + truncate_to_int(F * 100.0@S@) + B.

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, string.int_to_string(use(make(N))) ++ " | " ++
    string.int_to_string(use(make(N + 1))), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
