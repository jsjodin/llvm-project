% MANUAL: Terms
% Higher-order terms whose closure is a variable or a parenthesized field
% selection.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[2, 10]
% OUTPUT-NOT:{{.}}

:- module run_higher_order_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type box ---> box(fn :: (func(int) = int)).

main(!IO) :-
    F = (func(X) = X * 2),
    B = box(F),
    io.write([F(1), (B ^ fn)(5)], !IO), io.nl(!IO).
