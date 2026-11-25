% MANUAL: Conditional expressions
% Both forms of conditional expression, nested, and a condition's some
% [X] scoping over the then part.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{"no four", 2, 30, 2}
% OUTPUT-NOT:{{.}}

:- module run_conditional_expressions.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    L = [1, 2, 3],
    A = ( if list.member(4, L) then "four" else "no four" ),
    B = ( list.member(2, L) -> 2 ; 0 ),
    C = ( if some [X] (X = 3, list.member(X, L)) then X * 10 else 0 ),
    D = ( if A = "four" then 1 else if B = 2 then 2 else 3 ),
    io.write({A, B, C, D}, !IO), io.nl(!IO).
