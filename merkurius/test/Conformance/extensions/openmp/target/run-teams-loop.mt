% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% A target teams distribute parallel do loop runs its iterations across a
% device's teams and their threads. A value carried from one iteration to
% the next is named in a reduction clause, and its partial results are
% combined across threads and teams; an array threaded through the loop is
% updated in place, each iteration its own elements, and comes back. It is a
% template, run for float and float32, whose array is packed 4 bytes to an
% element.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/teams_loop_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float.d/run_teams_loop_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float.d/run_teams_loop_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float.d/run_teams_loop_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float.d/run_teams_loop_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:{20, -50, 50}
% OUTPUT-FLOAT-NEXT:1.0
% OUTPUT-FLOAT-NEXT:1.0
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float32.d/run_teams_loop_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float32.d/run_teams_loop_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float32.d/run_teams_loop_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float32.d/run_teams_loop_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:{20, -50, 50}
% OUTPUT-FLOAT32-NEXT:1.0
% OUTPUT-FLOAT32-NEXT:1.0
% OUTPUT-FLOAT32-NOT:{{.}}
:- module run_teams_loop.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, @TM@, int.

:- pred stats(int::in, int::out, int::out, int::out) is det.
stats(N, Sum, Lo, Hi) :-
    %$omp target teams distribute parallel do reduction((+) : Sum)
    %$omp& reduction(min : Lo) reduction(max : Hi)
    ( for(I, 1, N), fromto(0, S0, S1, Sum), fromto(max_int, L0, L1, Lo),
        fromto(min_int, H0, H1, Hi) do
        V = (I * 37) mod 101 - 50,
        S1 = S0 + V,
        L1 = int.min(L0, V),
        H1 = int.max(H0, V)
    ).

:- pred normalize(array(@T@)::array_di, array(@T@)::array_uo) is det.
normalize(!A) :-
    N = array.size(!.A),
    %$omp target teams distribute parallel do reduction((+) : Sum)
    ( for(I, 0, N - 1), fromto(0.0@S@, S0, S1, Sum) do
        S1 = S0 + !.A ^ elem(I)
    ),
    %$omp target teams distribute parallel do
    ( for(I, 0, N - 1) do
        !A ^ elem(I) := !.A ^ elem(I) / Sum
    ).

main(!IO) :-
    stats(100000, Sum, Lo, Hi),
    io.write({Sum, Lo, Hi}, !IO), io.nl(!IO),
    array.init(4096, 0.5@S@, A0),
    normalize(A0, A),
    io.write_@T@(A ^ elem(0) * 4096.0@S@, !IO), io.nl(!IO),
    io.write_@T@(A ^ elem(4095) * 4096.0@S@, !IO), io.nl(!IO).
