% MANUAL: Determinism categories
% MANUAL: Creating higher-order terms
% MANUAL: Predicate and function type declarations
% Interaction: nondet and semidet code, in polymorphic procedures that carry
% the values, in closures that make, capture and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_closure_polymorphic_determinism_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_closure_polymorphic_determinism_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_closure_polymorphic_determinism_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: evens 2, first over 2 3 / evens 2, first over 2 3 | evens 2,4, first over 2 3 | evens 2, first over 2 3 / evens 2, first over 2 3 | evens 2,4, first over 2 3 | evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3
% OUTPUT-NEXT:2: evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3
% OUTPUT-NEXT:3: evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4, first over 2 3 / evens 2,4, first over 2 3 | evens 2,4,6, first over 2 3 | evens 2,4,6, first over 2 3 / evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3
% OUTPUT-NEXT:4: evens 2,4,6, first over 2 3 / evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3 | evens 2,4,6, first over 2 3 / evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3 | evens 2,4,6, first over 2 3 / evens 2,4,6, first over 2 3 | evens 2,4,6,8, first over 2 3
% OUTPUT-NOT:{{.}}
:- module nested_closure_polymorphic_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_closure_polymorphic_determinism_lib, string.

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
