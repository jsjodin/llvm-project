% MANUAL: Existentially typed data types
% DESIGN: Loops
% MANUAL: Creating higher-order terms
% Interaction: an existentially typed constructor and its dictionary, in
% closures that make, capture and use the values, in a do loop that makes,
% gathers and carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_do_loop_closure_existential_types_translation.m
% MODULES: Inputs/nested_do_loop_closure_existential_types_lib.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_do_loop_closure_existential_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_do_loop_closure_existential_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: string z / string z | string z | int 14 | int 14 / int 14 | int 14 | string zzz | last string zzzzzzzzzzzzzzzzzzzzz / string zzzzzzzzzzzzzzzzzzzzz | string zzzzzzzzzzzzzzzzzzzzz | int 154
% OUTPUT-NEXT:2: int 14 / int 14 | int 14 | string zzz | string zzz / string zzz | string zzz | int 28 | last int 154 / int 154 | int 154 | string zzzzzzzzzzzzzzzzzzzzzzz
% OUTPUT-NEXT:3: string zzz / string zzz | string zzz | int 28 | int 28 / int 28 | int 28 | string zzzzz | last string zzzzzzzzzzzzzzzzzzzzzzz / string zzzzzzzzzzzzzzzzzzzzzzz | string zzzzzzzzzzzzzzzzzzzzzzz | int 168
% OUTPUT-NEXT:4: int 28 / int 28 | int 28 | string zzzzz | string zzzzz / string zzzzz | string zzzzz | int 42 | last int 168 / int 168 | int 168 | string zzzzzzzzzzzzzzzzzzzzzzzzz
% OUTPUT-NOT:{{.}}
:- module nested_do_loop_closure_existential_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_closure_existential_types_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Values made in a loop's iterations and gathered by a fromto, and a
    % value carried from one iteration to the next.
:- func run(int) = string.

run(N) = S :-
    ( foreach(K, [N, N + 1]), fromto([], Ss0, Ss1, Ss) do
        V = make_nested(K),
        Ss1 = [use_nested(V) | Ss0]
    ),
    ( for(I, 1, 2), fromto(make_nested(N), V0, V1, Last) do
        ( if I > 0 then V1 = make_nested(N + I * 10) else V1 = V0 )
    ),
    S = string.join_list(" | ", list.reverse(Ss)) ++ " | last " ++ use_nested(Last).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
