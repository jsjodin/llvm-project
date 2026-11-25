% MANUAL: Field access functions
% DESIGN: Loops
% DESIGN: Loops / Under a parallel directive
% Interaction: field access and update functions, in a parallel loop's
% iterations, which make and use the values, in a do loop that makes,
% gathers and carries the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_do_loop_parallel_do_field_access_translation.m
% MODULES: Inputs/nested_do_loop_parallel_do_field_access_lib.m
% FILES: Inputs/nested_do_loop_parallel_do_field_access_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_field_access_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_field_access_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_field_access_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_do_loop_parallel_do_field_access_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: owner1 15 10 / owner1 15 10 | owner2 25 20 | owner3 35 30 | owner4 45 40 | owner2 25 20 / owner2 25 20 | owner3 35 30 | owner4 45 40 | owner5 55 50 | last owner21 215 210 / owner21 215 210 | owner22 225 220 | owner23 235 230 | owner24 245 240
% OUTPUT-NEXT:2: owner2 25 20 / owner2 25 20 | owner3 35 30 | owner4 45 40 | owner5 55 50 | owner3 35 30 / owner3 35 30 | owner4 45 40 | owner5 55 50 | owner6 65 60 | last owner22 225 220 / owner22 225 220 | owner23 235 230 | owner24 245 240 | owner25 255 250
% OUTPUT-NEXT:3: owner3 35 30 / owner3 35 30 | owner4 45 40 | owner5 55 50 | owner6 65 60 | owner4 45 40 / owner4 45 40 | owner5 55 50 | owner6 65 60 | owner7 75 70 | last owner23 235 230 / owner23 235 230 | owner24 245 240 | owner25 255 250 | owner26 265 260
% OUTPUT-NEXT:4: owner4 45 40 / owner4 45 40 | owner5 55 50 | owner6 65 60 | owner7 75 70 | owner5 55 50 / owner5 55 50 | owner6 65 60 | owner7 75 70 | owner8 85 80 | last owner24 245 240 / owner24 245 240 | owner25 255 250 | owner26 265 260 | owner27 275 270
% OUTPUT-NOT:{{.}}
:- module nested_do_loop_parallel_do_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_do_loop_parallel_do_field_access_lib, string.

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
