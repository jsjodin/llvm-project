% The feature and inner context of run-layouts.mt.
:- module nested_polymorphic_nondet_layouts_lib.
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
:- import_module int, list, solutions.

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
