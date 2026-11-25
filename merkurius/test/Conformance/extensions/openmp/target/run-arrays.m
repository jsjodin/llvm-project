% DESIGN: Target regions
% Arrays of scalars cross a target region: an array the goal only reads
% goes in, and one it threads, di to uo, is updated in place on the device
% and comes back, the same array after the goal. Arrays of narrow integers
% and of an enum's constants hold their elements unboxed, and cross as
% arrays of words do. The goal's procedures run on the device too.
% MERKC-FLAGS: -fopenmp
%
% REQUIRES: libomp
% RUN: merkc -O0 -fopenmp %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopenmp %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:4.0
% OUTPUT-NEXT:4.0 -8.0 1.0 32.0
% OUTPUT-NEXT:0.5 1.5 2.0
% OUTPUT-NEXT:251u8 255u8 0u8 194u8
% OUTPUT-NEXT:green blue red
% OUTPUT-NOT:{{.}}
:- module run_arrays.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, float, int, list, string, uint8.

:- type colour ---> red ; green ; blue.

:- pred sum(array(float)::array_ui, int::in, float::in, float::out) is det.
sum(A, I, S0, S) :-
    ( if I < 0 then
        S = S0
    else
        sum(A, I - 1, S0 + A ^ unsafe_elem(I), S)
    ).

:- pred scale(int::in, float::in, array(float)::array_di,
    array(float)::array_uo) is det.
scale(I, K, !A) :-
    ( if I < 0 then
        true
    else
        !A ^ unsafe_elem(I) := K * !.A ^ unsafe_elem(I),
        scale(I - 1, K, !A)
    ).

:- pred bump(int::in, array(uint8)::array_di, array(uint8)::array_uo) is det.
bump(I, !B) :-
    ( if I < 0 then
        true
    else
        !B ^ unsafe_elem(I) := !.B ^ unsafe_elem(I) + 250u8,
        bump(I - 1, !B)
    ).

:- func next(colour) = colour.
next(red) = green.
next(green) = blue.
next(blue) = red.

:- pred rotate(int::in, array(colour)::array_di, array(colour)::array_uo)
    is det.
rotate(I, !C) :-
    ( if I < 0 then
        true
    else
        !C ^ unsafe_elem(I) := next(!.C ^ unsafe_elem(I)),
        rotate(I - 1, !C)
    ).

% Reads A, updates B, both in one region.
:- pred region(array(float)::array_ui, array(float)::array_di,
    array(float)::array_uo, float::out) is det.
region(A, !B, S) :-
    %$omp target
    (
        sum(A, array.size(A) - 1, 0.0, S),
        scale(array.size(!.B) - 1, S, !B)
    ).

:- pred narrow(array(uint8)::array_di, array(uint8)::array_uo,
    array(colour)::array_di, array(colour)::array_uo) is det.
narrow(!B, !C) :-
    %$omp target
    (
        bump(array.size(!.B) - 1, !B),
        rotate(array.size(!.C) - 1, !C)
    ).

:- pred show(array(T)::array_ui, io::di, io::uo) is det.
show(A, !IO) :-
    Strings = list.map(string.string, array.to_list(A)),
    io.write_string(string.join_list(" ", Strings), !IO),
    io.nl(!IO).

main(!IO) :-
    A = array.from_list([0.5, 1.5, 2.0]),
    B0 = array.from_list([1.0, -2.0, 0.25, 8.0]),
    region(A, B0, B, S),
    io.write_float(S, !IO), io.nl(!IO),
    show(B, !IO),
    show(A, !IO),
    narrow(array.from_list([1u8, 5u8, 6u8, 200u8]), N,
        array.from_list([red, green, blue]), C),
    show(N, !IO),
    show(C, !IO).
