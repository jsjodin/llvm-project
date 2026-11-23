% MANUAL: Insts modes and mode definitions
% The manual's insts and modes: a named bound inst and its ---> form, an
% inst for one type constructor, a mode written inst >> inst and named, the
% (+) and (-) modes Prolog fans may define, parametric insts and modes,
% and the library's in(I) and out(I).
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:"v1"
% OUTPUT-NEXT:"w2"
% OUTPUT-NEXT:7
% OUTPUT-NEXT:5
% OUTPUT-NEXT:6
% OUTPUT-NEXT:yes(7)
% OUTPUT-NEXT:6
% OUTPUT-NEXT:[1, 0, 2]
% OUTPUT-NOT:{{.}}

:- module run_manual_examples.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, maybe, string.

:- inst maybeskel
    --->    no
    ;       yes(ground).
:- inst maybeskel_eq == bound(no ; yes(ground)).
:- inst some_int for maybe/1
    --->    yes(ground).

:- type operation
    --->    lookup(int, string)
    ;       set(int, string).
:- inst set_request for operation/0
    --->    set(ground, ground).
:- mode create_request == free >> set_request.
:- mode satisfy_request == set_request >> ground.
:- mode create_request_out == out(set_request).

:- mode (+) == in.
:- mode (-) == out.
:- mode m == ground >> ground.

:- inst listskel(Inst) for list/1
    --->    []
    ;       [Inst | listskel(Inst)].

:- pred make_set(int::in, operation::create_request) is det.
make_set(K, set(K, "v")).

:- pred make_set_out(int::in, operation::create_request_out) is det.
make_set_out(K, set(K, "w")).

:- pred satisfy(operation::satisfy_request, string::out) is det.
satisfy(set(K, V), V ++ string.int_to_string(K)).

:- pred plus(int::(+), int::(+), int::(-)) is det.
plus(X, Y, X + Y).

:- pred same(int::m, int::out) is det.
same(X, X).

:- pred yes_value(maybe(int)::in(some_int), int::out) is det.
yes_value(yes(X), X).

:- pred skel(maybe(int)::in(maybeskel), maybe(int)::out(maybeskel_eq)) is det.
skel(M, M).

:- pred sum(list(int)::in(listskel(ground)), int::out) is det.
sum([], 0).
sum([X | Xs], X + S) :-
    sum(Xs, S).

:- pred lengths(list(list(int))::in(listskel(listskel(ground))),
    list(int)::out) is det.
lengths([], []).
lengths([L | Ls], [list.length(L) | Ns]) :-
    lengths(Ls, Ns).

main(!IO) :-
    make_set(1, R),
    satisfy(R, S),
    io.write_line(S, !IO),
    make_set_out(2, R2),
    satisfy(R2, S2),
    io.write_line(S2, !IO),
    plus(3, 4, P),
    io.write_line(P, !IO),
    same(5, Q),
    io.write_line(Q, !IO),
    yes_value(yes(6), Y),
    io.write_line(Y, !IO),
    skel(yes(7), M),
    io.write_line(M, !IO),
    sum([1, 2, 3], Sum),
    io.write_line(Sum, !IO),
    lengths([[1], [], [2, 3]], Ns),
    io.write_line(Ns, !IO).
