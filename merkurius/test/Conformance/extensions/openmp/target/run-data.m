% DESIGN: Target regions
% A target data region keeps arrays on the device for its goal, so that the
% target regions in it, even in procedures that it calls, find them there.
% A state variable's version at the goal is mapped, tofrom by default, and
% after the region the host has the device's elements. The goal may do I/O
% on the host, unlike a target region's.
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
% OUTPUT:in the region
% OUTPUT-NEXT:1000000
% OUTPUT-NEXT:500500
% OUTPUT-NEXT:1000
% OUTPUT-NOT:{{.}}
:- module run_data.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred add(int::in, int::in, array(int)::array_di, array(int)::array_uo)
    is det.
add(I, K, !A) :-
    ( if I < 0 then
        true
    else
        !A ^ unsafe_elem(I) := !.A ^ unsafe_elem(I) + K * I,
        add(I - 1, K, !A)
    ).

:- pred total(array(int)::array_ui, int::in, int::in, int::out) is det.
total(A, I, S0, S) :-
    ( if I < 0 then
        S = S0
    else
        total(A, I - 1, S0 + A ^ unsafe_elem(I), S)
    ).

:- pred step(int::in, array(int)::array_di, array(int)::array_uo) is det.
step(K, !A) :-
    %$omp target
    add(array.size(!.A) - 1, K, !A).

:- pred sum(array(int)::array_ui, int::out) is det.
sum(A, S) :-
    %$omp target
    total(A, array.size(A) - 1, 0, S).

main(!IO) :-
    some [!A] (
        array.init(1000, 1, !:A),
        %$omp target data map(tofrom: A)
        (
            step(2, !A),
            sum(!.A, S1),
            io.write_string("in the region\n", !IO),
            step(-1, !A),
            sum(!.A, S2)
        ),
        io.write_int(S1, !IO), io.nl(!IO),
        io.write_int(S2, !IO), io.nl(!IO),
        io.write_int(!.A ^ elem(999), !IO), io.nl(!IO)
    ).
