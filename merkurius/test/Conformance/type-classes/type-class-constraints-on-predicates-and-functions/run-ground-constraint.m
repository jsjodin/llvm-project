% MANUAL: Type class constraints on predicates and functions
% The manual says each constraint on a predicate must contain a variable,
% but mmc accepts c(int), and the program runs (design document, "Where
% mmc and the manual disagree").
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:0
% OUTPUT-NOT:{{.}}

:- module run_ground_constraint.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- typeclass c(T) where [
    func f(T) = int
].

:- instance c(int) where [
    f(_) = 0
].

:- pred p(int::out) is det <= c(int).
p(0).

main(!IO) :-
    p(X),
    io.write_int(X, !IO).
