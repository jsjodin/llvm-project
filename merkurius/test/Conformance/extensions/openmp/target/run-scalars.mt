% DESIGN: Target regions
% DESIGN: Sized floating point types
% A target region runs its goal on a device. Scalars cross it: ints, uints
% and the sized integers, floats, chars, enums and a type of one functor of
% one argument, which is its argument. The values the goal uses from outside
% go in, mapped by copy, and the variables it binds come out. To mmc the
% directive is a comment, so the program is plain Mercury. It is a template,
% run for float and float32.
% MERKC-FLAGS: -fopenmp
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fopenmp %t.float.d/run_scalars_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %t.float.d/run_scalars_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 %t.float.d/run_scalars_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_scalars_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:41
% OUTPUT-FLOAT-NEXT:42
% OUTPUT-FLOAT-NEXT:44
% OUTPUT-FLOAT-NEXT:5000000000000
% OUTPUT-FLOAT-NEXT:0.75
% OUTPUT-FLOAT-NEXT:b
% OUTPUT-FLOAT-NEXT:red
% OUTPUT-FLOAT-NEXT:2.5
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fopenmp %t.float32.d/run_scalars_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %t.float32.d/run_scalars_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 %t.float32.d/run_scalars_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_scalars_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:41
% OUTPUT-FLOAT32-NEXT:42
% OUTPUT-FLOAT32-NEXT:44
% OUTPUT-FLOAT32-NEXT:5000000000000
% OUTPUT-FLOAT32-NEXT:0.75
% OUTPUT-FLOAT32-NEXT:b
% OUTPUT-FLOAT32-NEXT:red
% OUTPUT-FLOAT32-NEXT:2.5
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_scalars.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module char, @TM@, int, int64, uint, uint8.

:- type colour ---> red ; green ; blue.
:- type meters ---> meters(@T@).

:- func next(colour) = colour.
next(red) = green.
next(green) = blue.
next(blue) = red.

:- pred region(int::in, uint::in, uint8::in, int64::in, @T@::in, char::in,
    colour::in, meters::in, int::out, uint::out, uint8::out, int64::out,
    @T@::out, char::out, colour::out, meters::out) is det.
region(I, U, B, L, F, C, K, M, I1, U1, B1, L1, F1, C1, K1, M1) :-
    %$omp target
    (
        I1 = I * 3 - 1,
        U1 = U * 2u,
        B1 = B + 200u8,
        L1 = L * 1_000_000i64,
        F1 = F / 4.0@S@,
        C1 = char.det_from_int(char.to_int(C) + 1),
        K1 = next(K),
        M = meters(X),
        M1 = meters(X * 2.0@S@)
    ).

main(!IO) :-
    region(14, 21u, 100u8, 5_000_000i64, 3.0@S@, 'a', blue, meters(1.25@S@),
        I, U, B, L, F, C, K, meters(M)),
    io.write_int(I, !IO), io.nl(!IO),
    io.write_uint(U, !IO), io.nl(!IO),
    io.write_uint8(B, !IO), io.nl(!IO),
    io.write_int64(L, !IO), io.nl(!IO),
    io.write_@T@(F, !IO), io.nl(!IO),
    io.write_char(C, !IO), io.nl(!IO),
    io.write(K, !IO), io.nl(!IO),
    io.write_@T@(M, !IO), io.nl(!IO).
