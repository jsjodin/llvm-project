% The feature of run-type-classes.m, as another module.
:- module another_module_type_classes_lib.
:- interface.

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

:- implementation.
:- import_module int, list, string.

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
