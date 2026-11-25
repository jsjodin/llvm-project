% MANUAL: Goals
% promise_equivalent_solution_sets with an arbitrary goal inside.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:positive
% OUTPUT-NOT:{{.}}

:- module run_equivalent_solution_sets.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred any_of(int::out) is multi.
any_of(1).
any_of(2).

main(!IO) :-
    promise_equivalent_solution_sets [S] (
        arbitrary [A] any_of(A),
        S = ( if A > 0 then "positive" else "not" )
    ),
    io.write_string(S, !IO), io.nl(!IO).
