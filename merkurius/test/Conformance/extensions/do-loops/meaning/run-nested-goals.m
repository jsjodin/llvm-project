% DESIGN: Loops / Meaning
% A loop may be any goal: in a lambda's body, where it is translated with
% the lambda's body, and under a negation, `not' and `\+' alike, where it
% is translated in place, and the negation succeeds where the loop fails.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_goals_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:6
% OUTPUT-NEXT:all odd
% OUTPUT-NEXT:not all odd
% OUTPUT-NOT:{{.}}
:- module run_nested_goals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    Sum = (func(Xs) = S :-
        ( foreach(X, Xs), fromto(0, S0, S1, S) do S1 = S0 + X )),
    io.write_int(Sum([1, 2, 3]), !IO),
    io.nl(!IO),
    L = [1, 3, 5],
    ( if not ( foreach(Y, L) do Y mod 2 = 1 ) then
        io.write_string("not all odd\n", !IO)
    else
        io.write_string("all odd\n", !IO)
    ),
    ( if \+ ( foreach(Z, [2 | L]) do Z mod 2 = 1 ) then
        io.write_string("not all odd\n", !IO)
    else
        io.write_string("all odd\n", !IO)
    ).
