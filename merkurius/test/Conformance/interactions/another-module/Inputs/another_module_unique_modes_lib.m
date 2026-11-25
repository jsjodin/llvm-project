% The feature of run-unique-modes.m, as another module.
:- module another_module_unique_modes_lib.
:- interface.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module array, int, list, string.

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
