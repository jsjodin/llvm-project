% MANUAL: Purity levels
% Clauses for particular modes of a predicate promised pure.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[2, 9]
% OUTPUT-NOT:{{.}}

:- module run_mode_specific_clauses_promised.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- pred f(int, int).
:- mode f(in, out) is det.
:- mode f(out, in) is det.
:- pragma promise_pure(f/2).
f(X::in, Y::out) :- Y = X + 1.
f(X::out, Y::in) :- X = Y - 1.

main(!IO) :- f(1, A), f(B, 10), io.write([A, B], !IO), io.nl(!IO).
