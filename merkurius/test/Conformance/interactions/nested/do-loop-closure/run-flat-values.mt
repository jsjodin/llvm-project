% MANUAL: Discriminated unions
% DESIGN: Loops
% MANUAL: Creating higher-order terms
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in closures
% that make, capture and use the values, in a do loop that makes, gathers
% and carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_do_loop_closure_flat_values_translation.mt
% MODULES: Inputs/nested_do_loop_closure_flat_values_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_do_loop_closure_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_do_loop_closure_flat_values_float.m %t.float.d/nested_do_loop_closure_flat_values_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_do_loop_closure_flat_values_float.m %t.float.d/nested_do_loop_closure_flat_values_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | last red 2100 -21 21000 5.25 / red 2100 -21 21000 5.25 | red 2100 -21 21000 5.25 | green 2200 -22 22000 5.5
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | last green 2200 -22 22000 5.5 / green 2200 -22 22000 5.5 | green 2200 -22 22000 5.5 | blue 2300 -23 23000 5.75
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | last blue 2300 -23 23000 5.75 / blue 2300 -23 23000 5.75 | blue 2300 -23 23000 5.75 | red 2400 -24 24000 6.0
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | last red 2400 -24 24000 6.0 / red 2400 -24 24000 6.0 | red 2400 -24 24000 6.0 | green 2500 -25 25000 6.25
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_do_loop_closure_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_do_loop_closure_flat_values_float32.m %t.float32.d/nested_do_loop_closure_flat_values_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_do_loop_closure_flat_values_float32.m %t.float32.d/nested_do_loop_closure_flat_values_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | last red 2100 -21 21000 5.25 / red 2100 -21 21000 5.25 | red 2100 -21 21000 5.25 | green 2200 -22 22000 5.5
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | last green 2200 -22 22000 5.5 / green 2200 -22 22000 5.5 | green 2200 -22 22000 5.5 | blue 2300 -23 23000 5.75
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | last blue 2300 -23 23000 5.75 / blue 2300 -23 23000 5.75 | blue 2300 -23 23000 5.75 | red 2400 -24 24000 6.0
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 | last red 2400 -24 24000 6.0 / red 2400 -24 24000 6.0 | red 2400 -24 24000 6.0 | green 2500 -25 25000 6.25
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_do_loop_closure_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_closure_flat_values_lib, string.

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
