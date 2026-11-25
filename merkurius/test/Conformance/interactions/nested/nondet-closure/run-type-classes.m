% MANUAL: Type classes
% MANUAL: Determinism categories
% MANUAL: Creating higher-order terms
% Interaction: a type class's methods called through a constraint, in
% closures that make, capture and use the values, in a nondet goal that
% makes the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_nondet_closure_type_classes_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_nondet_closure_type_classes_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_nondet_closure_type_classes_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: rect=12, square=1 / rect=12, square=1 | rect=12, square=1 | square=16, square=1 | rect=2, square=1 / rect=2, square=1 | rect=2, square=1 | square=4, square=1
% OUTPUT-NEXT:2: rect=12, square=1 / rect=12, square=1 | rect=12, square=1 | square=16, square=1
% OUTPUT-NEXT:3: rect=12, square=1 / rect=12, square=1 | rect=12, square=1 | square=16, square=1 | rect=30, square=1 / rect=30, square=1 | rect=30, square=1 | square=36, square=1
% OUTPUT-NEXT:4: rect=30, square=1 / rect=30, square=1 | rect=30, square=1 | square=36, square=1
% OUTPUT-NOT:{{.}}
:- module nested_nondet_closure_type_classes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_nondet_closure_type_classes_lib, solutions, string.

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
