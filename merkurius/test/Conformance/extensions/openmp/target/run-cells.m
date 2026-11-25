% DESIGN: Target regions
% A term of a type of one constructor whose arguments are all scalars is a
% cell of scalars, which crosses into a target region as an array does, by
% its words, and the goal reads its arguments there.
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
% OUTPUT:10.5
% OUTPUT-NEXT:26
% OUTPUT-NEXT:8
% OUTPUT-NOT:{{.}}
:- module run_cells.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

:- type vec ---> vec(float, float, float).
:- type span ---> span(int, int).

:- func dot(vec, vec) = float.
dot(vec(A, B, C), vec(X, Y, Z)) = A * X + B * Y + C * Z.

:- pred count(int::in, int::in, int::in, int::out) is det.
count(I, Hi, N0, N) :-
    ( if I > Hi then
        N = N0
    else
        count(I + 1, Hi, ( if I mod 3 = 0 then N0 + 1 else N0 ), N)
    ).

:- pred region(vec::in, vec::in, span::in, float::out, int::out, int::out)
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
    region(vec(1.0, 2.0, 3.0), vec(0.5, -1.0, 4.0), span(-5, 20), D, W, N),
    io.write_float(D, !IO), io.nl(!IO),
    io.write_int(W, !IO), io.nl(!IO),
    io.write_int(N, !IO), io.nl(!IO).
