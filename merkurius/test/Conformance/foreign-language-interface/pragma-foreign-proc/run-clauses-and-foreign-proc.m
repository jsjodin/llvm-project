% MANUAL: pragma foreign_proc
% A predicate of one mode with both clauses and a foreign_proc for C: the
% foreign_proc is its definition when compiling to C.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:101
% OUTPUT-NOT:{{.}}

:- module run_clauses_and_foreign_proc.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.
:- pred inc(int::in, int::out) is det.
:- pragma foreign_proc("C", inc(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X + 100;").
inc(X, X + 1).

main(!IO) :- inc(1, A), io.write_int(A, !IO), io.nl(!IO).
