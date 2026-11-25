% MANUAL: Trace goals
% Calls of procedures with trace goals whose outputs are unused, or that
% have none.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:d 3
% OUTPUT-NEXT:e 4
% OUTPUT-NEXT:d 5
% OUTPUT-NEXT:10
% OUTPUT-NOT:{{.}}

:- module run_calls_kept.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred d(int::in, int::out) is det.
d(X, Y) :- trace [io(!IO)] ( io.format("d %d\n", [i(X)], !IO) ), Y = X * 2.
:- pred e(int::in) is det.
e(X) :- trace [io(!IO)] ( io.format("e %d\n", [i(X)], !IO) ).
main(!IO) :- d(3, _), e(4), d(5, Y), io.write_int(Y, !IO), io.nl(!IO).
