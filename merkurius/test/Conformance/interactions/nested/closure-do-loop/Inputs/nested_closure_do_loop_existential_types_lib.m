% The feature and inner context of run-existential-types.m.
:- module nested_closure_do_loop_existential_types_lib.
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
:- import_module int, list.

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

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.

inner_run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).
