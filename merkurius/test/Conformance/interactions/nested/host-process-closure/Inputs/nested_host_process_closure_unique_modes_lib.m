% The feature and inner context of run-unique-modes.m.
:- module nested_host_process_closure_unique_modes_lib.
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

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.

inner_run(N) = S :-
    Make = (func(K) = make(K)),
    V = apply(Make, N),
    Use = ((func) = use(V)),
    Both = list.map((func(K) = use(make(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).
