% The feature and inner context of run-open-types.m.
:- module nested_polymorphic_nondet_open_types_lib_translation.
:- interface.
:- import_module list, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int, solutions.

make(N) = (1 .. N) ++ [N * 10].

use(L) =
    ( if L = [A, B | Rest] then
        string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        "short"
    ).

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
