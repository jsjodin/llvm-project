% The feature and inner context of run-unique-modes.m.
:- module nested_closure_typeclass_method_unique_modes_lib.
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

:- typeclass runner(T) where [
    func produce(T, int) = fdata,
    func consume(T, fdata) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make(N),
    consume(plain, V) = use(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make(N + K),
    consume(offset(_), V) = "offset " ++ use(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).


inner_run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).
