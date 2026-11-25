% DESIGN: Open terms (Prolog-style variables)
% MANUAL: Determinism categories
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in a nondet goal that makes the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/nondet_open_types_translation.m
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 10 rest  | 1 2 rest 3,30
% OUTPUT-NEXT:2: 1 2 rest 3,30
% OUTPUT-NEXT:3: 1 2 rest 3,30 | 1 2 rest 3,4,5,50
% OUTPUT-NEXT:4: 1 2 rest 3,4,5,50
% OUTPUT-NOT:{{.}}
:- module nondet_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, solutions, string.

:- type fdata == list(int).

:- func make(int) = fdata.
:- func use(fdata) = string.

    % Built as a difference list, whose hole is closed with a ground tail.
make(N) = L :-
    make_open(1 .. N, Front, Hole),
    promise_pure (
        if close(Front, Hole, [N * 10], L0) then
            L = L0
        else
            L = []
    ).

:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.
:- pragma promise_pure(close/4).

close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    impure from_open(Front, List).

    % Taken apart by unifying it with an open pattern, whose holes the
    % unification binds; then the hole of a second pattern is bound to the
    % first's rest, and closed.
use(L) = S :-
    ( if take_apart(L, A, B, Rest) then
        S = string.int_to_string(A) ++ " " ++ string.int_to_string(B) ++
            " rest " ++ string.join_list(",",
                list.map(string.int_to_string, Rest))
    else
        S = "short"
    ).

:- pred take_apart(list(int)::in, int::out, int::out, list(int)::out)
    is semidet.
:- pragma promise_pure(take_apart/4).

take_apart(L, A, B, Rest) :-
    P = [OA, OB | ORest],
    P = to_open(L),
    impure from_open(OA, A),
    impure from_open(OB, B),
    impure from_open(ORest, Rest).

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
