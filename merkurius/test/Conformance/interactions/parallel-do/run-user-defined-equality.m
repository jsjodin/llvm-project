% MANUAL: User-defined equality and comparison
% DESIGN: Loops / Under a parallel directive
% Interaction: a type with an equality of its own, in a parallel loop's
% iterations, which make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/parallel_do_user_defined_equality_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp %s %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %s %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %s -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 members, not equal to [1, 2, 3] | 2 members, not equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:2: 2 members, not equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:3: 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:4: 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] | 7 members, not equal to [1, 2, 3]
% OUTPUT-NOT:{{.}}
:- module parallel_do_user_defined_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, string.

    % A set as a list in any order, equal by its members.
:- type fdata
    --->    set(list(int))
    where equality is same_members.

:- pred same_members(fdata::in, fdata::in) is semidet.

:- func make(int) = fdata.
:- func use(fdata) = string.

same_members(A, B) :-
    promise_equivalent_solutions [SA, SB] (
        A = set(LA),
        B = set(LB),
        list.sort_and_remove_dups(LA, SA),
        list.sort_and_remove_dups(LB, SB)
    ),
    SA = SB.

make(N) = set(list.reverse(1 .. N) ++ [1]).

use(D) = S :-
    % Built first, so that the unification compares two values, by the
    % type's equality.
    Expected = set([3, 2, 1]),
    ( if D = Expected then
        Equal = "equal to [1, 2, 3]"
    else
        Equal = "not equal to [1, 2, 3]"
    ),
    promise_equivalent_solutions [Count] (
        D = set(L),
        list.sort_and_remove_dups(L, Members),
        Count = list.length(Members)
    ),
    S = string.int_to_string(Count) ++ " members, " ++ Equal.

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make(N + I),
        !A ^ unsafe_elem(I) := use(V)
    ).

:- func run(int) = string.

run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
