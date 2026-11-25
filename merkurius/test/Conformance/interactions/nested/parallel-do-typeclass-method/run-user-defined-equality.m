% MANUAL: User-defined equality and comparison
% DESIGN: Loops / Under a parallel directive
% MANUAL: Type classes
% Interaction: a type with an equality of its own, in instance methods that
% make and use the values, in a parallel loop's iterations, which make and
% use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_parallel_do_typeclass_method_user_defined_equality_translation.m
% MODULES: Inputs/nested_parallel_do_typeclass_method_user_defined_equality_lib.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_typeclass_method_user_defined_equality_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_parallel_do_typeclass_method_user_defined_equality_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_typeclass_method_user_defined_equality_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_parallel_do_typeclass_method_user_defined_equality_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 members, not equal to [1, 2, 3] / 1 members, not equal to [1, 2, 3] | offset 2 members, not equal to [1, 2, 3] | 2 members, not equal to [1, 2, 3] / 2 members, not equal to [1, 2, 3] | offset 3 members, equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] / 3 members, equal to [1, 2, 3] | offset 4 members, not equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:2: 2 members, not equal to [1, 2, 3] / 2 members, not equal to [1, 2, 3] | offset 3 members, equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] / 3 members, equal to [1, 2, 3] | offset 4 members, not equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] / 5 members, not equal to [1, 2, 3] | offset 6 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:3: 3 members, equal to [1, 2, 3] / 3 members, equal to [1, 2, 3] | offset 4 members, not equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] / 5 members, not equal to [1, 2, 3] | offset 6 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] / 6 members, not equal to [1, 2, 3] | offset 7 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:4: 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] / 5 members, not equal to [1, 2, 3] | offset 6 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] / 6 members, not equal to [1, 2, 3] | offset 7 members, not equal to [1, 2, 3] | 7 members, not equal to [1, 2, 3] / 7 members, not equal to [1, 2, 3] | offset 8 members, not equal to [1, 2, 3]
% OUTPUT-NOT:{{.}}
:- module nested_parallel_do_typeclass_method_user_defined_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module array, int, list, nested_parallel_do_typeclass_method_user_defined_equality_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use_nested makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make_nested(N + I),
        !A ^ unsafe_elem(I) := use_nested(V)
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
