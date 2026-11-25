% MANUAL: Field access functions
% MANUAL: Type classes
% Interaction: field access and update functions, in instance methods that
% make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: owner1 15 10 | offset owner2 25 20
% OUTPUT-NEXT:2: owner2 25 20 | offset owner3 35 30
% OUTPUT-NEXT:3: owner3 35 30 | offset owner4 45 40
% OUTPUT-NEXT:4: owner4 45 40 | offset owner5 55 50
% OUTPUT-NOT:{{.}}
:- module typeclass_method_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type fdata
    --->    account(
                owner :: string,
                balance :: int,
                history :: list(int)
            ).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = account("owner" ++ string.int_to_string(N), N * 10, []).

use(A0) = S :-
    A1 = A0 ^ balance := A0 ^ balance + 5,
    A = A1 ^ history := [A0 ^ balance | A1 ^ history],
    S = A ^ owner ++ " " ++ string.int_to_string(A ^ balance) ++ " " ++
        string.join_list(",", list.map(string.int_to_string, A ^ history)).

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
