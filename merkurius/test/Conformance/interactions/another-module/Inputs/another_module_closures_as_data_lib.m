% The feature of run-closures-as-data.m, as another module.
:- module another_module_closures_as_data_lib.
:- interface.

    % Functions as data, which capture the argument of make.
:- type fdata
    --->    fns(func(int) = int, func(int) = string).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- implementation.
:- import_module int, string.

make(N) = fns((func(X) = X * N + 1),
    (func(X) = string.int_to_string(X) ++ "/" ++ string.int_to_string(N))).

use(fns(F, G)) = string.int_to_string(apply(F, 10)) ++ " " ++ G(F(2)).
