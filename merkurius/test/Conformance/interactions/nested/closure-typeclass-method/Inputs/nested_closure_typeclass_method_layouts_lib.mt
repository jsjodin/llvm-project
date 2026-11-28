% The feature and inner context of run-layouts.mt.
:- module nested_closure_typeclass_method_layouts_lib.
:- interface.
:- import_module @TM@, char, string.

    % More functors with arguments than a pointer has tags, constants, and
    % arguments of every kind of word.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, string)
    ;       three(@T@, fdata, int)
    ;       four(char, int, int, int)
    ;       five(string)
    ;       six(int)
    ;       seven(fdata, fdata)
    ;       eight(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.

:- func inner_run(int) = string.

:- implementation.
:- import_module int.

make(N) = seven(three(@FROM_INT@(N) / 2.0@S@, one(N), N * 3),
    ( if N mod 2 = 0 then
        eight(c0)
    else if N mod 3 = 0 then
        four('x', N, -N, N * N)
    else
        two(N, "s" ++ string.int_to_string(N))
    )).

use(c0) = "c0".
use(c1) = "c1".
use(one(I)) = "one(" ++ string.int_to_string(I) ++ ")".
use(two(I, S)) = "two(" ++ string.int_to_string(I) ++ ", " ++ S ++ ")".
use(three(F, D, I)) = "three(" ++ string.@T@_to_string(F) ++ ", " ++
    use(D) ++ ", " ++ string.int_to_string(I) ++ ")".
use(four(C, A, B, D)) = "four(" ++ string.char_to_string(C) ++ ", " ++
    string.int_to_string(A + B + D) ++ ")".
use(five(S)) = "five(" ++ S ++ ")".
use(six(I)) = "six(" ++ string.int_to_string(I) ++ ")".
use(seven(A, B)) = "seven(" ++ use(A) ++ ", " ++ use(B) ++ ")".
use(eight(D)) = "eight(" ++ use(D) ++ ")".

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
