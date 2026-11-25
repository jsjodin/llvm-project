% DESIGN: Open terms (Prolog-style variables)
% DESIGN: Loops
% DESIGN: Loops / Under a parallel directive
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in a parallel loop's iterations, which make and use the
% values, in a do loop that makes, gathers and carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp -fopen-types
% TRANSLATION: Inputs/nested_do_loop_parallel_do_open_types_translation.m
% MODULES: Inputs/nested_do_loop_parallel_do_open_types_lib.m
% FILES: Inputs/nested_do_loop_parallel_do_open_types_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -fopen-types -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_open_types_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -fopen-types -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_open_types_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -fopen-types -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_open_types_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopen-types -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_open_types_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 10 rest  / 1 10 rest  | 1 2 rest 20 | 1 2 rest 3,30 | 1 2 rest 3,4,40 | 1 2 rest 20 / 1 2 rest 20 | 1 2 rest 3,30 | 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,210 / 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,210 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,220 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240
% OUTPUT-NEXT:2: 1 2 rest 20 / 1 2 rest 20 | 1 2 rest 3,30 | 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | 1 2 rest 3,30 / 1 2 rest 3,30 | 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,6,60 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,220 / 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,220 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,250
% OUTPUT-NEXT:3: 1 2 rest 3,30 / 1 2 rest 3,30 | 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,6,60 | 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,6,60 | 1 2 rest 3,4,5,6,7,70 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230 / 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,230 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,250 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,260
% OUTPUT-NEXT:4: 1 2 rest 3,4,40 / 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,6,60 | 1 2 rest 3,4,5,6,7,70 | 1 2 rest 3,4,5,50 / 1 2 rest 3,4,5,50 | 1 2 rest 3,4,5,6,60 | 1 2 rest 3,4,5,6,7,70 | 1 2 rest 3,4,5,6,7,8,80 | last 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240 / 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,240 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,250 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,260 | 1 2 rest 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,270
% OUTPUT-NOT:{{.}}
:- module nested_do_loop_parallel_do_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_parallel_do_open_types_lib, string.

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
