% DESIGN: Target regions
% DESIGN: Sized floating point types
% A term of a type of one constructor whose arguments are all scalars is a
% cell of scalars, which crosses into a target region as an array does, by
% its words, and the goal reads its arguments there. It is a template, run
% for float and float32, whose cells hold their floats in words.
% MERKC-FLAGS: -fopenmp
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fopenmp %t.float.d/run_cells_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %t.float.d/run_cells_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 %t.float.d/run_cells_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float.d/run_cells_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:10.5
% OUTPUT-FLOAT-NEXT:26
% OUTPUT-FLOAT-NEXT:8
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fopenmp %t.float32.d/run_cells_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %t.float32.d/run_cells_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 %t.float32.d/run_cells_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 %t.float32.d/run_cells_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:10.5
% OUTPUT-FLOAT32-NEXT:26
% OUTPUT-FLOAT32-NEXT:8
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_cells.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, int.

:- type vec ---> vec(@T@, @T@, @T@).
:- type span ---> span(int, int).

:- func dot(vec, vec) = @T@.
dot(vec(A, B, C), vec(X, Y, Z)) = A * X + B * Y + C * Z.

:- pred count(int::in, int::in, int::in, int::out) is det.
count(I, Hi, N0, N) :-
    ( if I > Hi then
        N = N0
    else
        count(I + 1, Hi, ( if I mod 3 = 0 then N0 + 1 else N0 ), N)
    ).

:- pred region(vec::in, vec::in, span::in, @T@::out, int::out, int::out)
    is det.
region(U, V, S, D, W, N) :-
    %$omp target
    (
        D = dot(U, V),
        S = span(Lo, Hi),
        W = Hi - Lo + 1,
        count(Lo, Hi, 0, N)
    ).

main(!IO) :-
    region(vec(1.0@S@, 2.0@S@, 3.0@S@), vec(0.5@S@, -1.0@S@, 4.0@S@), span(-5, 20), D, W, N),
    io.write_@T@(D, !IO), io.nl(!IO),
    io.write_int(W, !IO), io.nl(!IO),
    io.write_int(N, !IO), io.nl(!IO).
