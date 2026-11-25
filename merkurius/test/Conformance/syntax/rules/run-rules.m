% MANUAL: Rules
% Predicate and function rules.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:{4, 6}
% OUTPUT-NOT:{{.}}

:- module run_rules.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred double(int::in, int::out) is det.
double(X, Y) :- Y = X * 2.
:- func triple(int) = int.
triple(X) = Y :- Y = X * 3.

main(!IO) :- double(2, A), io.write({A, triple(2)}, !IO), io.nl(!IO).
