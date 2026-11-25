% The feature and inner context of run-existential-types.m.
:- module nested_closure_do_loop_existential_types_lib_translation.
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


inner_run(N) = string.join_list(" | ",
    list.map((func(K) = use(make(K))), [N, N + 1])) ++ " | last " ++
    use(make(N + 20)).
