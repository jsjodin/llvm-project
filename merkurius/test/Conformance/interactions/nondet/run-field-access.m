% MANUAL: Field access functions
% MANUAL: Determinism categories
% Interaction: field access and update functions, in a nondet goal that
% makes the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: owner1 15 10 | owner3 35 30
% OUTPUT-NEXT:2: owner3 35 30
% OUTPUT-NEXT:3: owner3 35 30 | owner5 55 50
% OUTPUT-NEXT:4: owner5 55 50
% OUTPUT-NOT:{{.}}
:- module nondet_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions, string.

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

    % Values made in a nondet goal, each solution's in turn.
:- pred made(int::in, string::out) is nondet.

made(N, S) :-
    list.member(K, [N, N + 1, N + 2]),
    K mod 2 = 1,
    V = make(K),
    S = use(V).

:- func run(int) = string.

run(N) = S :-
    solutions(made(N), Ss),
    S = string.join_list(" | ", Ss).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
