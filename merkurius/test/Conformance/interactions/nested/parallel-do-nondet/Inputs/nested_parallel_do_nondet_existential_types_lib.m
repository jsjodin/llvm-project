% The feature and inner context of run-existential-types.m.
:- module nested_parallel_do_nondet_existential_types_lib.
:- interface.
:- import_module string.

:- typeclass showable(T) where [
    func show(T) = string
].

:- instance showable(int).
:- instance showable(string).

    % A value of a hidden type, with its dictionary.
:- type fdata
    --->    some [T] boxed(T) => showable(T).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, list, solutions.

:- instance showable(int) where [
    show(I) = "int " ++ string.int_to_string(I)
].
:- instance showable(string) where [
    show(S) = "string " ++ S
].

make(N) = D :-
    ( if N mod 2 = 0 then
        D = 'new boxed'(N * 7)
    else
        D = 'new boxed'(string.duplicate_char('z', N))
    ).

use(boxed(X)) = show(X).

    % Values made in a nondet goal, each solution's in turn.
:- pred made(int::in, string::out) is nondet.

made(N, S) :-
    list.member(K, [N, N + 1, N + 2]),
    K mod 2 = 1,
    V = make(K),
    S = use(V).


inner_run(N) = S :-
    solutions(made(N), Ss),
    S = string.join_list(" | ", Ss).
