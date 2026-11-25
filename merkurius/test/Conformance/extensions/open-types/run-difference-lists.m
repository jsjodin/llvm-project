% DESIGN: Open terms (Prolog-style variables)
% The design's two examples: a difference list built through its open
% tail and closed with a ground one, and Prolog's app(A-B, B-C, A-C) on
% difference lists closed into an ordinary list, which a caller in ordinary
% Mercury takes apart. Closing a hole that is bound already fails. mmc
% rejects building a difference list; the oracle is a translation into
% Prolog.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/difference_lists_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:[1, 2, 3]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[1, 2, 3, 4, 5]
% OUTPUT-NEXT:[]
% OUTPUT-NOT:{{.}}
:- module run_difference_lists.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, pair.

:- pred make_open(list(int)::in,
    open(list(int))::oa, open(list(int))::oa) is det.

make_open([], T, T).
make_open([X | Xs], [X | R], T) :-
    make_open(Xs, R, T).

    % Fill the hole with Tail and return the ground list.
:- pred close(open(list(int))::ia, open(list(int))::ia,
    list(int)::in, list(int)::out) is semidet.
:- pragma promise_pure(close/4).

close(Front, Hole, Tail, List) :-
    Hole = to_open(Tail),
    impure from_open(Front, List).

:- type dlist == open(pair(list(int), list(int))).   % Front - Hole

:- pred append_dl(dlist::ia, dlist::ia, dlist::oa) is semidet.
append_dl(A - B, B - C, A - C).

    % Close the hole with [] and return an ordinary list.
:- pred dl_to_list(dlist::ia, list(int)::out) is semidet.
:- pragma promise_pure(dl_to_list/2).
dl_to_list(F - [], L) :-
    impure from_open(F, L).

:- pred test(list(int)::out) is semidet.
test(L) :-
    D1 = [1, 2 | H1] - H1,
    D2 = [3 | H2] - H2,
    append_dl(D1, D2, D),
    dl_to_list(D, L).

:- pred first(int::out) is semidet.
first(X) :-
    test(L),
    L = [X | _].

    % The same hole closed twice: the second time it is bound to [4, 5]
    % already, which [6] does not unify with.
:- pred twice(list(int)::out, list(int)::out) is semidet.
twice(L1, L2) :-
    make_open([1, 2, 3], Front, Hole),
    close(Front, Hole, [4, 5], L1),
    promise_pure (
        if close(Front, Hole, [6], L) then L2 = L else L2 = []
    ).

main(!IO) :-
    make_open([1, 2, 3], Front, Hole),
    promise_pure (
        if close(Front, Hole, [4, 5], List) then
            io.write_line(List, !IO)
        else
            io.write_string("hole already bound\n", !IO)
    ),
    ( if test(L) then io.write_line(L, !IO) else io.write_string("no\n", !IO) ),
    ( if first(X) then
        io.write_line(X, !IO)
    else
        io.write_string("no\n", !IO)
    ),
    ( if twice(A, B) then
        io.write_line(A, !IO), io.write_line(B, !IO)
    else
        io.write_string("no\n", !IO)
    ).
