% DESIGN: Loops / Iterators
% fromto(First, In, Out, Last) is an accumulator from First to Last. When
% Last has variables it is the final value; when it is ground, as [] or
% stop, it ends the loop: fromto(L, [X | Xs], Xs, []) traverses a list, and
% fromto(continue, continue, C, stop) loops until C is stop.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/fromto_translation.m
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:5.0
% OUTPUT-NEXT:[2, 1, 3]
% OUTPUT-NEXT:243
% OUTPUT-NOT:{{.}}
:- module run_fromto.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int, list.

:- type go ---> continue ; stop.

main(!IO) :-
    ( for(I, 1, 4), fromto(0.0, S0, S1, Sum) do S1 = S0 + float(I) / 2.0 ),
    io.write_float(Sum, !IO), io.nl(!IO),
    ( fromto([3, 1, 2], [X | Xs], Xs, []), fromto([], R0, R1, Reversed) do
        R1 = [X | R0]
    ),
    io.write(Reversed, !IO), io.nl(!IO),
    ( fromto(continue, continue, C, stop), fromto(1, P0, P1, Power) do
        P1 = P0 * 3,
        C = ( if P1 > 100 then stop else continue )
    ),
    io.write_int(Power, !IO), io.nl(!IO).
