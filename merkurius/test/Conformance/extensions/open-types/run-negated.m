% DESIGN: Open terms (Prolog-style variables) / Checking rules
% A negation or if-then-else condition that may bind a nonlocal open
% variable is in a promise_pure scope, the rule Mercury applies to solver
% types. A binding made in it is undone when it fails, and so is one made
% in a negation that succeeds: \+ X = 1 with X unbound fails, as in Prolog,
% and leaves X unbound. A disjunct that binds and then fails leaves its
% bindings to be undone before the next. Bindings that a condition makes
% and keeps are seen by the then part.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/negated_translation.pl
%
% RUN: merkc -O0 -fopen-types %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:negated_unbound failed, X: _1
% OUTPUT-NEXT:negated_bound succeeded, X: 2
% OUTPUT-NEXT:condition_undone else, L: [_1, _2]
% OUTPUT-NEXT:condition_kept then, L: [5, _1]
% OUTPUT-NEXT:disjunct_undone, L: [_1, _2, 3]
% OUTPUT-NEXT:nested then, L: [_1, 8]
% OUTPUT-NEXT:aliased else, L: [_1, _1]
% OUTPUT-NOT:{{.}}
:- module run_negated.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, open, string.

:- pred show(string::in, open(T)::ia, io::di, io::uo) is det.

show(Label, X, !IO) :-
    promise_pure (
        impure S = open.string(X),
        io.format("%s: %s\n", [s(Label), s(S)], !IO)
    ).

    % \+ X = 1 with X unbound: the unification succeeds, so the negation
    % fails, and X is unbound after it.
:- pred negated_unbound(open(int)::oa, string::out) is det.

negated_unbound(X, Answer) :-
    X = X0,
    promise_pure (
        if \+ X0 = to_open(1) then Answer = "succeeded" else Answer = "failed"
    ).

    % \+ X = 1 with X bound to 2 succeeds and binds nothing.
:- pred negated_bound(open(int)::oa, string::out) is det.

negated_bound(X, Answer) :-
    X = to_open(2),
    promise_pure (
        if \+ X = to_open(1) then Answer = "succeeded" else Answer = "failed"
    ).

    % A condition that binds the list's first hole, then fails on the
    % second: the else part sees both unbound.
:- pred condition_undone(open(list(int))::oa, string::out) is det.

condition_undone(L, Answer) :-
    L = [A, B],
    promise_pure (
        if A = to_open(1), B = to_open(2), A = B then
            Answer = "then"
        else
            Answer = "else"
    ).

    % A condition that binds and succeeds: the then part sees the binding,
    % and so does the caller.
:- pred condition_kept(open(list(int))::oa, string::out) is det.

condition_kept(L, Answer) :-
    L = [A, _],
    promise_pure (
        if A = to_open(5) then Answer = "then" else Answer = "else"
    ).

    % The first disjunct binds A, B and C and then fails; the second sees
    % them unbound and binds C only. The disjunction is multi, as each
    % disjunct binds the list's holes, so the first solution is committed
    % to, written as a string.
:- pred disjunct_undone(string::out) is det.

disjunct_undone(S) :-
    promise_pure (
        promise_equivalent_solutions [S] (
            L = [A, B, C] : open(list(int)),
            ( A = to_open(1), B = to_open(1), C = to_open(1), A = to_open(9)
            ; C = to_open(3)
            ),
            impure S = open.string(L)
        )
    ).

    % A negation inside a condition, whose goal binds A and then fails, so
    % that it succeeds: its binding is undone, and the condition's own
    % binding of B is kept.
:- pred nested(open(list(int))::oa, string::out) is det.

nested(L, Answer) :-
    L = [A, B],
    promise_pure (
        if \+ ( A = to_open(7), A = to_open(8) ), B = to_open(8) then
            Answer = "then"
        else
            Answer = "else"
    ).

    % One variable reached two ways: binding it through one alias in a
    % failed condition is undone for the other too.
:- pred aliased(open(list(int))::oa, string::out) is det.

aliased(L, Answer) :-
    L = [A, B],
    A = B,
    promise_pure (
        if A = to_open(4), B = to_open(5) then
            Answer = "then"
        else
            Answer = "else"
    ).

main(!IO) :-
    negated_unbound(X1, A1),
    io.format("negated_unbound %s, ", [s(A1)], !IO),
    show("X", X1, !IO),
    negated_bound(X2, A2),
    io.format("negated_bound %s, ", [s(A2)], !IO),
    show("X", X2, !IO),
    condition_undone(L3, A3),
    io.format("condition_undone %s, ", [s(A3)], !IO),
    show("L", L3, !IO),
    condition_kept(L4, A4),
    io.format("condition_kept %s, ", [s(A4)], !IO),
    show("L", L4, !IO),
    disjunct_undone(S5),
    io.format("disjunct_undone, L: %s\n", [s(S5)], !IO),
    nested(L6, A6),
    io.format("nested %s, ", [s(A6)], !IO),
    show("L", L6, !IO),
    aliased(L7, A7),
    io.format("aliased %s, ", [s(A7)], !IO),
    show("L", L7, !IO).
