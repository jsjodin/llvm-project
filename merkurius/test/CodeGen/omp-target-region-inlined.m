% A call in a target region, with a pointer live across it, in a procedure that
% is inlined into a caller that has a pointer of its own live. The call is still
% not a safepoint, as a region's thread is pinned.
%
% RUN: merkc -O2 -fopenmp %libomp-flags %s -o %t && %libomp-env %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.sequential && %t.sequential | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% REQUIRES: libomp

:- module omp_target_region_inlined.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int.

:- pred line(string::in, int::in, io::di, io::uo) is det.
line(Text, N, !IO) :-
    io.write_string(Text, !IO),
    io.write_int(N, !IO),
    io.nl(!IO).

:- pred region_in_procedure(array(int)::in, int::out, io::di, io::uo) is det.
region_in_procedure(A, N, !IO) :-
    %$omp target
    (
        B = array.init(3, 7),
        line("first element: ", A ^ elem(0), !IO),
        N = B ^ elem(1) + A ^ elem(1)
    ).

main(!IO) :-
    Kept = array.init(2, 5),
    region_in_procedure(Kept, Sum, !IO),
    line("after the procedure: ", Sum + Kept ^ elem(0), !IO).

% OUTPUT:first element: 5
% OUTPUT-NEXT:after the procedure: 17
