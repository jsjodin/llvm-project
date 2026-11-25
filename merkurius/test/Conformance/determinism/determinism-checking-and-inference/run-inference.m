% MANUAL: Determinism checking and inference
% Local procedures without determinism annotations have it inferred. A
% unification with a known top function symbol is det; the manual's
% switches, through an alias, existential quantification and a nested
% disjunction; a multi goal without visible outputs is det; and the
% negation rules. A declaration weaker than the inferred determinism is
% allowed.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:10
% OUTPUT-NEXT:positive
% OUTPUT-NEXT:3
% OUTPUT-NEXT:{2, 0, 5, 10}
% OUTPUT-NEXT:["not fail is det", "not true is failure", "not positive"]
% OUTPUT-NEXT:weaker succeeded
% OUTPUT-NOT:{{.}}

:- module run_inference.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- type shape
    --->    f
    ;       g
    ;       h(int).

% Inferred det.
:- func twice(int) = int.
:- mode twice(in) = out.
twice(X) = X * 2.

% Inferred semidet.
:- pred positive(int::in).
positive(X) :-
    X > 0.

:- pred known(int::out) is det.
known(Y) :-
    X = h(3),
    X = h(Y).

:- pred empty(int::out) is det.
empty(0).

:- pred nonempty(int::in, list(int)::in, int::out) is det.
nonempty(H, T, H + list.length(T)).

:- pred plain(list(int)::in, int::out) is det.
plain(L, Out) :-
    (
        L = [],
        empty(Out)
    ;
        L = [H | T],
        nonempty(H, T, Out)
    ).

:- pred aliased(list(int)::in, int::out) is det.
aliased(L, Out) :-
    (
        L = [],
        empty(Out)
    ;
        K = L,
        K = [H | T],
        nonempty(H, T, Out)
    ).

:- pred quantified(list(int)::in, int::out) is det.
quantified(L, Out) :-
    (
        Out = 1,
        L = []
    ;
        some [H, T] (
            L = [H | T],
            nonempty(H, T, Out)
        )
    ).

:- pred nested(shape::in, int::out) is det.
nested(X, Out) :-
    (
        X = f,
        Out = 0
    ;
        Y = X,
        (
            Y = g,
            Intermediate = 42
        ;
            Z = Y,
            Z = h(Arg),
            Intermediate = Arg + 1
        ),
        Out = Intermediate * 2
    ).

:- pred some_list(list(int)::out) is multi.
some_list([]).
some_list([1]).

:- pred no_outputs is det.
no_outputs :-
    some_list(_).

:- pred negations(int::in, list(string)::out) is det.
negations(X, L) :-
    ( if not fail then A = "not fail is det" else A = "?" ),
    ( if not true then B = "?" else B = "not true is failure" ),
    ( if not X > 0 then C = "not positive" else C = "positive" ),
    L = [A, B, C].

:- pred weaker(int::in, int::out) is nondet.
weaker(X, X + 1).

main(!IO) :-
    io.write_line(twice(5), !IO),
    ( if positive(3) then
        io.write_string("positive\n", !IO)
    else
        io.write_string("not positive\n", !IO)
    ),
    known(K),
    io.write_line(K, !IO),
    plain([1, 2], P),
    aliased([], A),
    quantified([5], Q),
    nested(h(4), N),
    io.write_line({P, A, Q, N}, !IO),
    no_outputs,
    negations(-1, Ns),
    io.write_line(Ns, !IO),
    ( if weaker(1, _) then
        io.write_string("weaker succeeded\n", !IO)
    else
        io.write_string("none\n", !IO)
    ).
