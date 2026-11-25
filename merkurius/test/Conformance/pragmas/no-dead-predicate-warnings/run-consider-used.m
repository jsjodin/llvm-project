% MANUAL: No dead predicate warnings
% A consider_used pragma for a predicate no one calls, which calls
% another.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:ok
% OUTPUT-NOT:{{.}}

:- module run_consider_used.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred unused(int::out) is det.
:- pragma consider_used(unused/1).
unused(X) :- helper(X).
:- pred helper(int::out) is det.
helper(2).

main(!IO) :- io.write_string("ok\n", !IO).
