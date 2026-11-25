% MANUAL: Syntax for explicit type quantifiers
% The manual says universal quantifiers must precede existential ones, but
% mmc accepts some before all, and several quantifiers of each kind ("Where
% mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1
% OUTPUT-NEXT:"two"
% OUTPUT-NOT:{{.}}

:- module run_quantifier_order.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- some [T2] all [T3] pred foo(T1, T2, T3).
:- mode foo(in, out, in) is det.
foo(_, 1, _).

:- all [T1] some [T2] all [T3] pred bar(T1, T2, T3).
:- mode bar(in, out, in) is det.
bar(_, "two", _).

main(!IO) :-
    foo(1, X, 'c'),
    io.write_line(X, !IO),
    bar(1.5, Y, 'c'),
    io.write_line(Y, !IO).
