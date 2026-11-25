% MANUAL: Facts
% Predicate and function facts, with mode-qualified arguments.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{1, "red"}
% OUTPUT-NOT:{{.}}

:- module run_facts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred colour(string, int).
:- mode colour(in, out) is semidet.
:- mode colour(out, in) is semidet.
:- pragma promise_equivalent_clauses(colour/2).
colour("red"::in, 1::out).
colour("red"::out, 1::in).
:- func one = int.
one = 1.

main(!IO) :-
    ( if colour("red", N), colour(S, one) then io.write({N, S}, !IO) else true ),
    io.nl(!IO).
