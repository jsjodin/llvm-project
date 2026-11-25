% The feature and inner context of run-unique-modes.m.
:- module nested_polymorphic_nondet_unique_modes_lib.
:- interface.
:- import_module string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, int, list, solutions.

make(N) = N + 3.

use(Size) = string.join_list(" ",
    list.map(string.int_to_string, array.to_list(squares(0, A0)))) :-
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
