% MANUAL: User-defined equality and comparison
% MANUAL: Type classes
% DESIGN: Loops / Under a parallel directive
% Interaction: a type with an equality of its own, in a parallel loop's
% iterations, which make and use the values, in instance methods that make
% and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_typeclass_method_parallel_do_user_defined_equality_translation.m
% MODULES: Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib.m
% FILES: Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_typeclass_method_parallel_do_user_defined_equality_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 members, not equal to [1, 2, 3] / 1 members, not equal to [1, 2, 3] | 2 members, not equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | offset 2 members, not equal to [1, 2, 3] / 2 members, not equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:2: 2 members, not equal to [1, 2, 3] / 2 members, not equal to [1, 2, 3] | 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | offset 3 members, equal to [1, 2, 3] / 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:3: 3 members, equal to [1, 2, 3] / 3 members, equal to [1, 2, 3] | 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] | offset 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] | 7 members, not equal to [1, 2, 3]
% OUTPUT-NEXT:4: 4 members, not equal to [1, 2, 3] / 4 members, not equal to [1, 2, 3] | 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] | 7 members, not equal to [1, 2, 3] | offset 5 members, not equal to [1, 2, 3] / 5 members, not equal to [1, 2, 3] | 6 members, not equal to [1, 2, 3] | 7 members, not equal to [1, 2, 3] | 8 members, not equal to [1, 2, 3]
% OUTPUT-NOT:{{.}}
:- module nested_typeclass_method_parallel_do_user_defined_equality.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_typeclass_method_parallel_do_user_defined_equality_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

:- typeclass runner(T) where [
    func produce(T, int) = nested,
    func consume(T, nested) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make_nested(N),
    consume(plain, V) = use_nested(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make_nested(N + K),
    consume(offset(_), V) = "offset " ++ use_nested(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).

:- func run(int) = string.

run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
