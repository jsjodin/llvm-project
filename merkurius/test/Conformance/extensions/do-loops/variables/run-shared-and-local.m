% DESIGN: Loops / Variables
% A variable of the body that also occurs around the loop is shared by all
% iterations; every other one is local to each iteration, so loops one
% after another may use the same names, each for its own. A value leaves a
% loop through an iterator or a state variable.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/shared_and_local_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:[101, 104, 109]
% OUTPUT-NEXT:[-99, -98]
% OUTPUT-NOT:{{.}}
:- module run_shared_and_local.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    Offset = 100,
    ( for(I, 1, 3), foreach(Y, Ys) do T = I * I, Y = T + Offset ),
    ( for(I, 1, 2), foreach(Y, Zs) do T = I - Offset, Y = T ),
    io.write(Ys, !IO), io.nl(!IO),
    io.write(Zs, !IO), io.nl(!IO).
