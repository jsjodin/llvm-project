% DESIGN: Open terms (Prolog-style variables)
% MANUAL: Creating higher-order terms
% MANUAL: Type classes
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in instance methods that make and use the values, in closures
% that make, capture and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/nested_closure_typeclass_method_open_types_translation.m
% MODULES: Inputs/nested_closure_typeclass_method_open_types_lib.m
% FILES: Inputs/nested_closure_typeclass_method_open_types_lib_translation.m
%
% RUN: merkc -O0 -fopen-types -I %S/Inputs %s %S/Inputs/nested_closure_typeclass_method_open_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types -I %S/Inputs %s %S/Inputs/nested_closure_typeclass_method_open_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 10 rest  / 1 10 rest  | offset 1 2 rest 20 | 1 10 rest  / 1 10 rest  | offset 1 2 rest 20 | 1 2 rest 20 / 1 2 rest 20 | offset 1 2 rest 3,30
% OUTPUT-NEXT:2: 1 2 rest 20 / 1 2 rest 20 | offset 1 2 rest 3,30 | 1 2 rest 20 / 1 2 rest 20 | offset 1 2 rest 3,30 | 1 2 rest 3,30 / 1 2 rest 3,30 | offset 1 2 rest 3,4,40
% OUTPUT-NEXT:3: 1 2 rest 3,30 / 1 2 rest 3,30 | offset 1 2 rest 3,4,40 | 1 2 rest 3,30 / 1 2 rest 3,30 | offset 1 2 rest 3,4,40 | 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | offset 1 2 rest 3,4,5,50
% OUTPUT-NEXT:4: 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | offset 1 2 rest 3,4,5,50 | 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | offset 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,50 / 1 2 rest 3,4,5,50 | offset 1 2 rest 3,4,5,6,60
% OUTPUT-NOT:{{.}}
:- module nested_closure_typeclass_method_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_closure_typeclass_method_open_types_lib, string.

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
