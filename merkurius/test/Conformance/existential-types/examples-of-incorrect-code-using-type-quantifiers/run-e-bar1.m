% MANUAL: Examples of incorrect code using type quantifiers
% Clauses of an existentially typed predicate that all bind its type to
% int are correct.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NOT:{{.}}

:- module run_e_bar1.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- some [T] pred e_bar1(T).
:- mode e_bar1(out) is multi.
e_bar1(42).
e_bar1(42).
e_bar1(43).
    % ok (T = int)

main(!IO) :-
    e_bar1(X),
    io.write_line(X, !IO).
