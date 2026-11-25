% The feature and inner context of run-closures-as-data.m.
:- module nested_polymorphic_typeclass_method_closures_as_data_lib.
:- interface.
:- import_module string.

    % Functions as data, which capture the argument of make.
:- type fdata
    --->    fns(func(int) = int, func(int) = string).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int.

make(N) = fns((func(X) = X * N + 1),
    (func(X) = string.int_to_string(X) ++ "/" ++ string.int_to_string(N))).

use(fns(F, G)) = string.int_to_string(apply(F, 10)) ++ " " ++ G(F(2)).

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
