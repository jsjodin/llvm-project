% The feature and inner context of run-unique-modes.m.
:- module nested_try_do_loop_unique_modes_lib.
:- interface.
:- import_module string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module array, int, list.

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
