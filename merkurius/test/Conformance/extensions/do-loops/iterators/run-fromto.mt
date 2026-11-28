% DESIGN: Loops / Iterators
% DESIGN: Sized floating point types
% fromto(First, In, Out, Last) is an accumulator from First to Last. When
% Last has variables it is the final value; when it is ground, as [] or
% stop, it ends the loop: fromto(L, [X | Xs], Xs, []) traverses a list, and
% fromto(continue, continue, C, stop) loops until C is stop. It is a
% template, run for float and float32.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/fromto_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fdo-loops %t.float.d/run_fromto_float.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float.d/run_fromto_float.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:5.0
% OUTPUT-FLOAT-NEXT:[2, 1, 3]
% OUTPUT-FLOAT-NEXT:243
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops %t.float32.d/run_fromto_float32.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float32.d/run_fromto_float32.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:5.0
% OUTPUT-FLOAT32-NEXT:[2, 1, 3]
% OUTPUT-FLOAT32-NEXT:243
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_fromto.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int, list.

:- type go ---> continue ; stop.

main(!IO) :-
    ( for(I, 1, 4), fromto(0.0@S@, S0, S1, Sum) do
        S1 = S0 + @FROM_INT@(I) / 2.0@S@
    ),
    io.write_@T@(Sum, !IO), io.nl(!IO),
    ( fromto([3, 1, 2], [X | Xs], Xs, []), fromto([], R0, R1, Reversed) do
        R1 = [X | R0]
    ),
    io.write(Reversed, !IO), io.nl(!IO),
    ( fromto(continue, continue, C, stop), fromto(1, P0, P1, Power) do
        P1 = P0 * 3,
        C = ( if P1 > 100 then stop else continue )
    ),
    io.write_int(Power, !IO), io.nl(!IO).
