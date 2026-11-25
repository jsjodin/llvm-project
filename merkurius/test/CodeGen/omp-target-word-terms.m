% A term that is one word with no cell crosses a target region as its word,
% into it as well as out of it: a constant of a type of constants only, and
% a value of a type of one constructor of one argument, which is its
% argument, a scalar or another such term (design document, "Target
% regions"). The output is mmc's.
%
% RUN: merkc -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t.O2 && %libomp-env %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_word_terms.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module float, int.

:- type colour ---> red ; green ; blue.
:- type meters ---> meters(float).
:- type tagged ---> tagged(colour).

:- func next(colour) = colour.
next(red) = green.
next(green) = blue.
next(blue) = red.

:- pred region(colour::in, meters::in, tagged::in, colour::out, meters::out,
    tagged::out, int::out) is det.
region(K, M, T, K1, M1, T1, N) :-
    %$omp target
    (
        K1 = next(K),
        M = meters(X),
        M1 = meters(X * 2.0),
        T = tagged(C),
        T1 = tagged(next(next(C))),
        ( if K = blue then N = 1 else N = 0 )
    ).

main(!IO) :-
    region(blue, meters(1.25), tagged(green), K, meters(M), T, N),
    io.write(K, !IO), io.nl(!IO),
    io.write_float(M, !IO), io.nl(!IO),
    io.write(T, !IO), io.nl(!IO),
    io.write_int(N, !IO), io.nl(!IO).
% OUTPUT:red
% OUTPUT-NEXT:2.5
% OUTPUT-NEXT:tagged(red)
% OUTPUT-NEXT:1
% OUTPUT-NOT:{{.}}
