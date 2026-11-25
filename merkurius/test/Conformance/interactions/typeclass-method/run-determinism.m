% MANUAL: Determinism categories
% MANUAL: Type classes
% Interaction: nondet and semidet code, in instance methods that make and
% use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: evens 2, first over 2 3 | offset evens 2,4, first over 2 3
% OUTPUT-NEXT:2: evens 2,4, first over 2 3 | offset evens 2,4, first over 2 3
% OUTPUT-NEXT:3: evens 2,4, first over 2 3 | offset evens 2,4,6, first over 2 3
% OUTPUT-NEXT:4: evens 2,4,6, first over 2 3 | offset evens 2,4,6, first over 2 3
% OUTPUT-NOT:{{.}}
:- module typeclass_method_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = 1 .. N + 2.

use(L) = S :-
    solutions(even_member(L), Evens),
    ( if first_over(L, 2, F) then
        First = string.int_to_string(F)
    else
        First = "none"
    ),
    S = "evens " ++ string.join_list(",",
        list.map(string.int_to_string, Evens)) ++ ", first over 2 " ++ First.

:- pred even_member(list(int)::in, int::out) is nondet.

even_member(L, X) :-
    list.member(X, L),
    X mod 2 = 0.

:- pred first_over(list(int)::in, int::in, int::out) is semidet.

first_over([X | Xs], Limit, F) :-
    ( if X > Limit then
        F = X
    else
        first_over(Xs, Limit, F)
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

:- func run(int) = string.

run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
