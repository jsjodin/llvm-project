% DESIGN: Loops / Meaning
% A loop has the solutions of its translation: its modes and determinism
% are inferred as for a predicate without declarations, so a loop whose
% body has several solutions has several, in the translation's order.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nondet_body_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{{\[\[}}1, 2], [1, 20], [10, 2], [10, 20]]
% OUTPUT-NOT:{{.}}
:- module run_nondet_body.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, solutions.

:- pred choices(list(int)::out) is multi.
choices(L) :-
    ( foreach(X, [1, 2]), foreach(Y, L) do ( Y = X ; Y = X * 10 ) ).

main(!IO) :-
    solutions(choices, Ls),
    io.write(Ls, !IO), io.nl(!IO).
