% MANUAL: Equivalence types
% MANUAL: Creating higher-order terms
% DESIGN: Loops / Under a parallel directive
% Interaction: a type defined as another, generic one, in a parallel loop's
% iterations, which make and use the values, in closures that make, capture
% and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_closure_parallel_do_equivalence_types_translation.m
% MODULES: Inputs/nested_closure_parallel_do_equivalence_types_lib.m
% FILES: Inputs/nested_closure_parallel_do_equivalence_types_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_closure_parallel_do_equivalence_types_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_closure_parallel_do_equivalence_types_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_closure_parallel_do_equivalence_types_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_closure_parallel_do_equivalence_types_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1: n1 / 1: n1 | 2: n1 n2 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 1: n1 / 1: n1 | 2: n1 n2 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 2: n1 n2 / 2: n1 n2 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5
% OUTPUT-NEXT:2: 2: n1 n2 / 2: n1 n2 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 2: n1 n2 / 2: n1 n2 | 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6
% OUTPUT-NEXT:3: 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 7: n1 n2 n3 n4 n5 n6 n7
% OUTPUT-NEXT:4: 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 7: n1 n2 n3 n4 n5 n6 n7 | 4: n1 n2 n3 n4 / 4: n1 n2 n3 n4 | 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 7: n1 n2 n3 n4 n5 n6 n7 | 5: n1 n2 n3 n4 n5 / 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | 7: n1 n2 n3 n4 n5 n6 n7 | 8: n1 n2 n3 n4 n5 n6 n7 n8
% OUTPUT-NOT:{{.}}
:- module nested_closure_parallel_do_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_closure_parallel_do_equivalence_types_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.
:- func run(int) = string.

run(N) = S :-
    Make = (func(K) = make_nested(K)),
    V = apply(Make, N),
    Use = ((func) = use_nested(V)),
    Both = list.map((func(K) = use_nested(make_nested(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
