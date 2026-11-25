% MANUAL: Elimination of double negation
% An inequality in a negation, and all with an implication, make double
% negations that would otherwise not be mode-correct.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:a b c
% OUTPUT-NOT:{{.}}

:- module run_double_negation.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

main(!IO) :-
    L = [1, 2, 3],
    ( if not (X = 4, X \= 4) then io.write_string("a ", !IO) else true ),
    ( if all [Y] (list.member(Y, L) => Y < 10) then io.write_string("b", !IO) else true ),
    ( if not not list.member(2, L) then io.write_string(" c", !IO) else true ),
    io.nl(!IO).
