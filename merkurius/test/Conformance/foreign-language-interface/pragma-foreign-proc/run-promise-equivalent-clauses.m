% MANUAL: pragma foreign_proc
% A predicate of two modes with clauses and a foreign_proc for one mode,
% promised pure: the foreign_proc defines its mode, the clauses the other.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[2, 9]
% OUTPUT-NOT:{{.}}

:- module run_promise_equivalent_clauses.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.
:- pred inc(int, int).
:- pragma promise_pure(inc/2).
:- mode inc(in, out) is det.
:- mode inc(out, in) is det.
:- pragma foreign_proc("C", inc(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X + 1;").
inc(X, Y) :- X = Y - 1.

main(!IO) :- inc(1, A), inc(B, 10), io.write([A, B], !IO), io.nl(!IO).
