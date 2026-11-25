% DESIGN: Loops / Iterators
% All the iterators of a loop advance together, in lockstep; a nest is a
% loop in a loop's body, whose inner loop runs in full in each iteration
% of the outer one.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/lockstep_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[10, 41, 92]
% OUTPUT-NEXT:{{\[\[}}1, 2, 3], [2, 4, 6], [3, 6, 9]]
% OUTPUT-NOT:{{.}}
:- module run_lockstep.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    ( for(I, 0, 2), foreach(X, [10, 20, 30]), foreach(Y, [1, 2, 3]),
      foreach(Z, Zs) do
        Z = I + X * Y
    ),
    io.write(Zs, !IO), io.nl(!IO),
    ( for(I, 1, 3), foreach(Row, Table) do
        ( for(J, 1, 3), foreach(E, Row) do E = I * J )
    ),
    io.write(Table, !IO), io.nl(!IO).
