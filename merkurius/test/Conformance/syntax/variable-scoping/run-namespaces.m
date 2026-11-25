% MANUAL: Variable scoping
% Ordinary, type and inst variables with the same name are unrelated.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NOT:{{.}}

:- module run_namespaces.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred id(T::in(T), T::out(T)) is det.
id(T, T).
:- pred count(list(T)::in, int::out) is det.
count(T, N) :- N = list.length(T : list(T)).

main(!IO) :- id(3, X), count([1, 2, X], N), io.write_int(N, !IO), io.nl(!IO).
