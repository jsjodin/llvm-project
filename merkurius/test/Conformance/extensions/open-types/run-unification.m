% DESIGN: Open terms (Prolog-style variables) / Operations
% X = Y on open values is full unification: it binds variables on either
% side, one variable to another too, so that binding one later binds all
% that are the same, and fails on a constructor clash. X = f(Y1, ..., Yn)
% binds an unbound X to a new cell, or takes apart the cell X is bound to.
% Plain = omits the occurs check, as Prolog does: from_open rejects the
% cyclic term that can give, and unify_oc fails instead. A binding made in
% a condition that fails is undone. to_open of a ground term unifies with
% an open pattern. mmc has no open terms; the oracle is a translation
% into Prolog.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/unification_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[5, 5, _1]
% OUTPUT-NEXT:[7, 7, 7]
% OUTPUT-NEXT:clash
% OUTPUT-NEXT:p(1, _1)
% OUTPUT-NEXT:1
% OUTPUT-NEXT:cyclic, not closed
% OUTPUT-NEXT:unify_oc failed
% OUTPUT-NEXT:neither [_1, _2]
% OUTPUT-NEXT:1
% OUTPUT-NEXT:[2, 3]
% OUTPUT-NOT:{{.}}
:- module run_unification.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

:- type p ---> p(int, int).

:- pred show(open(T)::ia, io::di, io::uo) is det.
show(X, !IO) :-
    promise_pure ( impure open.write(X, !IO), io.nl(!IO) ).

    % The arguments are holes when they are passed, so the unifications are
    % of variables at run time.
:- pred same(open(int)::ia, open(int)::ia) is semidet.
same(X, X).

:- pred bind(open(int)::ia, int::in) is semidet.
bind(X, N) :-
    X = to_open(N).

    % A and B are made one variable, then B is bound: A is bound too.
:- pred aliasing(open(list(int))::oa) is semidet.
aliasing(L) :-
    L = [A, B, _],
    same(A, B),
    bind(B, 5).

:- pred chain(open(list(int))::oa) is semidet.
chain(L) :-
    L = [A, B, C],
    same(A, B),
    same(C, B),
    bind(C, 7).

:- pred clash(int::in, open(list(int))::oa) is semidet.
clash(N, L) :-
    L = [N | _],
    L = [2 | _].

    % X is bound to a new cell, then taken apart; the hole is left as it is.
:- pred structure(open(p)::oa, int::out) is det.
structure(X, A) :-
    X = p(1, _),
    X = p(A, _).

:- pred cyclic(string::out, string::out) is det.
cyclic(S1, S2) :-
    promise_pure (
        ( if X = [1 | X] then
            ( if impure from_open(X, L) then
                S1 = "closed " ++ string.string(L : list(int))
            else
                S1 = "cyclic, not closed"
            )
        else
            S1 = "not unified"
        ),
        ( if unify_oc(Y, [1 | Y] : open(list(int))) then
            S2 = "unify_oc unified"
        else
            S2 = "unify_oc failed"
        )
    ).

    % The condition binds the head, then fails: the binding is undone.
:- pred undone(open(list(int))::oa, string::out) is det.
undone(L, S) :-
    L = [A, _],
    promise_pure (
        if A = to_open(1), A = to_open(2) then S = "both" else S = "neither"
    ).

:- pred opened(open(int)::oa, list(int)::out) is semidet.
opened(A, T) :-
    to_open([1, 2, 3]) = [A | T0],
    promise_pure ( impure from_open(T0, T) ).

main(!IO) :-
    ( if aliasing(L1) then show(L1, !IO) else io.write_string("no\n", !IO) ),
    ( if chain(L2) then show(L2, !IO) else io.write_string("no\n", !IO) ),
    ( if clash(1, L3) then show(L3, !IO) else io.write_string("clash\n", !IO) ),
    structure(X, A), show(X, !IO), io.write_line(A, !IO),
    cyclic(S1, S2), io.write_string(S1 ++ "\n" ++ S2 ++ "\n", !IO),
    undone(L4, S4), io.write_string(S4 ++ " ", !IO), show(L4, !IO),
    ( if opened(B, T) then
        show(B, !IO), io.write_line(T, !IO)
    else
        io.write_string("no\n", !IO)
    ).
