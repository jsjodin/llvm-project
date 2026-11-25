% The translation of run-type-classes.m in host-process for mmc: the same
% feature in plain Mercury, which must print the same.
:- module host_process_type_classes_translation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- typeclass shape(T) where [
    func area(T) = int,
    func name(T) = string
].

:- type fdata
    --->    rect(int, int)
    ;       square(int).

:- instance shape(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- instance shape(fdata) where [
    area(rect(W, H)) = W * H,
    area(square(S)) = S * S,
    name(rect(_, _)) = "rect",
    name(square(_)) = "square"
].

make(N) = ( if N mod 2 = 0 then square(N) else rect(N, N + 1) ).

use(D) = describe([D, square(1)]).

    % Through a constraint, so that the method is called by its dictionary.
:- func describe(list(T)) = string <= shape(T).
:- pragma no_inline(describe/1).

describe(Shapes) = string.join_list(", ",
    list.map((func(S) = name(S) ++ "=" ++ string.int_to_string(area(S))),
        Shapes)).

:- pred run(int::in, string::out, io::di, io::uo) is det.

run(N, use(make(N)) ++ " | " ++ use(make(N + 1)), !IO).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    run(N, S, !IO),
    io.format("%d: %s\n", [i(N), s(S)], !IO).
