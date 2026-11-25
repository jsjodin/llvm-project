% MANUAL: Existentially typed data types
% MANUAL: Creating higher-order terms
% DESIGN: Loops
% Interaction: an existentially typed constructor and its dictionary, in a
% do loop that makes, gathers and carries the values, in closures that make,
% capture and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_closure_do_loop_existential_types_translation.m
% MODULES: Inputs/nested_closure_do_loop_existential_types_lib.m
% FILES: Inputs/nested_closure_do_loop_existential_types_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_closure_do_loop_existential_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_closure_do_loop_existential_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: string z / string z | int 14 | last string zzzzzzzzzzzzzzzzzzzzz | string z / string z | int 14 | last string zzzzzzzzzzzzzzzzzzzzz | int 14 / int 14 | string zzz | last int 154
% OUTPUT-NEXT:2: int 14 / int 14 | string zzz | last int 154 | int 14 / int 14 | string zzz | last int 154 | string zzz / string zzz | int 28 | last string zzzzzzzzzzzzzzzzzzzzzzz
% OUTPUT-NEXT:3: string zzz / string zzz | int 28 | last string zzzzzzzzzzzzzzzzzzzzzzz | string zzz / string zzz | int 28 | last string zzzzzzzzzzzzzzzzzzzzzzz | int 28 / int 28 | string zzzzz | last int 168
% OUTPUT-NEXT:4: int 28 / int 28 | string zzzzz | last int 168 | int 28 / int 28 | string zzzzz | last int 168 | string zzzzz / string zzzzz | int 42 | last string zzzzzzzzzzzzzzzzzzzzzzzzz
% OUTPUT-NOT:{{.}}
:- module nested_closure_do_loop_existential_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_closure_do_loop_existential_types_lib, string.

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
