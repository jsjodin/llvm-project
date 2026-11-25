% DESIGN: Open terms (Prolog-style variables)
% DESIGN: Loops
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in a do loop that makes, gathers and carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopen-types
% TRANSLATION: Inputs/do_loop_open_types_translation.m
%
% RUN: merkc -O0 -fdo-loops -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 10 rest  | 1 2 rest 20 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,210
% OUTPUT-NEXT:2: 1 2 rest 20 | 1 2 rest 3,30 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,220
% OUTPUT-NEXT:3: 1 2 rest 3,30 | 1 2 rest 3,4,40 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230
% OUTPUT-NEXT:4: 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240
% OUTPUT-NOT:{{.}}
:- module do_loop_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

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

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.
:- func run(int) = string.

run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make(K),
        Ss1 = [use(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make(N), V0, V1, Last) do
        ( if I > 0 then V1 = make(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use(Last).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
