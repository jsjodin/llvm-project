% MANUAL: Discriminated unions
% MANUAL: Determinism categories
% MANUAL: Type classes
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in instance
% methods that make and use the values, in a nondet goal that makes the
% values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_nondet_typeclass_method_flat_values_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_nondet_typeclass_method_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/nested_nondet_typeclass_method_flat_values_float.m %t.float.d/nested_nondet_typeclass_method_flat_values_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/nested_nondet_typeclass_method_flat_values_float.m %t.float.d/nested_nondet_typeclass_method_flat_values_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | offset blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:2: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:3: blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | offset red 600 -6 6000 1.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:4: blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | offset red 600 -6 6000 1.5
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_nondet_typeclass_method_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/nested_nondet_typeclass_method_flat_values_float32.m %t.float32.d/nested_nondet_typeclass_method_flat_values_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/nested_nondet_typeclass_method_flat_values_float32.m %t.float32.d/nested_nondet_typeclass_method_flat_values_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | offset blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:2: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:3: blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | offset red 600 -6 6000 1.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | offset green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:4: blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | offset red 600 -6 6000 1.5
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_nondet_typeclass_method_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_nondet_typeclass_method_flat_values_lib, solutions, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Values made in a nondet goal, each solution's in turn.
:- pred made(int::in, string::out) is nondet.

made(N, S) :-
    list.member(K, [N, N + 1, N + 2]),
    K mod 2 = 1,
    V = make_nested(K),
    S = use_nested(V).

:- func run(int) = string.

run(N) = S :-
    solutions(made(N), Ss),
    S = string.join_list(" | ", Ss).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
