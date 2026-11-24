% MANUAL: Backtrackable destructive update
% muo and mdi work on mostly unique values, which may be needed again on
% backtracking, as in a disjunction: add's input is used by both of its
% branches.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[11, 12]
% OUTPUT-NOT:{{.}}

:- module run_mostly_unique.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- import_module int, list, solutions.

:- type counter
    --->    counter(int).

:- pred make(counter::muo) is det.
make(counter(10)).

:- pred add(int::in, counter::mdi, counter::muo) is det.
add(I, counter(N), counter(N + I)).

:- pred choice(int::out) is multi.
choice(V) :-
    make(C0),
    ( I = 1 ; I = 2 ),
    add(I, C0, C),
    C = counter(V).

main(!IO) :-
    solutions(choice, Ls),
    io.write_line(Ls, !IO).
