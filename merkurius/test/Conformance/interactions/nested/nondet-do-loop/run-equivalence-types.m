% MANUAL: Equivalence types
% MANUAL: Determinism categories
% DESIGN: Loops
% Interaction: a type defined as another, generic one, in a do loop that
% makes, gathers and carries the values, in a nondet goal that makes the
% values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_nondet_do_loop_equivalence_types_translation.m
% MODULES: Inputs/nested_nondet_do_loop_equivalence_types_lib.m
% FILES: Inputs/nested_nondet_do_loop_equivalence_types_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_nondet_do_loop_equivalence_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_nondet_do_loop_equivalence_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1: n1 / 1: n1 | 2: n1 n2 | last 21: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 | 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | last 23: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23
% OUTPUT-NEXT:2: 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | last 23: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23
% OUTPUT-NEXT:3: 3: n1 n2 n3 / 3: n1 n2 n3 | 4: n1 n2 n3 n4 | last 23: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 | 5: n1 n2 n3 n4 n5 / 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | last 25: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25
% OUTPUT-NEXT:4: 5: n1 n2 n3 n4 n5 / 5: n1 n2 n3 n4 n5 | 6: n1 n2 n3 n4 n5 n6 | last 25: n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25
% OUTPUT-NOT:{{.}}
:- module nested_nondet_do_loop_equivalence_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_nondet_do_loop_equivalence_types_lib, solutions, string.

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
