% DESIGN: Target regions
% A target region runs its goal on a device. Scalars cross it: ints, uints
% and the sized integers, floats, chars, enums and a type of one functor of
% one argument, which is its argument. The values the goal uses from outside
% go in, mapped by copy, and the variables it binds come out. To mmc the
% directive is a comment, so the program is plain Mercury.
% MERKC-FLAGS: -fopenmp
%
% REQUIRES: libomp
% RUN: merkc -O0 -fopenmp %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:41
% OUTPUT-NEXT:42
% OUTPUT-NEXT:44
% OUTPUT-NEXT:5000000000000
% OUTPUT-NEXT:0.75
% OUTPUT-NEXT:b
% OUTPUT-NEXT:red
% OUTPUT-NEXT:2.5
% OUTPUT-NOT:{{.}}
:- module run_scalars.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, float, int, int64, uint, uint8.

:- type colour ---> red ; green ; blue.
:- type meters ---> meters(float).

:- func next(colour) = colour.
next(red) = green.
next(green) = blue.
next(blue) = red.

:- pred region(int::in, uint::in, uint8::in, int64::in, float::in, char::in,
    colour::in, meters::in, int::out, uint::out, uint8::out, int64::out,
    float::out, char::out, colour::out, meters::out) is det.
region(I, U, B, L, F, C, K, M, I1, U1, B1, L1, F1, C1, K1, M1) :-
    %$omp target
    (
        I1 = I * 3 - 1,
        U1 = U * 2u,
        B1 = B + 200u8,
        L1 = L * 1_000_000i64,
        F1 = F / 4.0,
        C1 = char.det_from_int(char.to_int(C) + 1),
        K1 = next(K),
        M = meters(X),
        M1 = meters(X * 2.0)
    ).

main(!IO) :-
    region(14, 21u, 100u8, 5_000_000i64, 3.0, 'a', blue, meters(1.25),
        I, U, B, L, F, C, K, meters(M)),
    io.write_int(I, !IO), io.nl(!IO),
    io.write_uint(U, !IO), io.nl(!IO),
    io.write_uint8(B, !IO), io.nl(!IO),
    io.write_int64(L, !IO), io.nl(!IO),
    io.write_float(F, !IO), io.nl(!IO),
    io.write_char(C, !IO), io.nl(!IO),
    io.write(K, !IO), io.nl(!IO),
    io.write_float(M, !IO), io.nl(!IO).
