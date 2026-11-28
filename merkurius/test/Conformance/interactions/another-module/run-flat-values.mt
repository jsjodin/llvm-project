% MANUAL: Discriminated unions
% MANUAL: Modules
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in the
% feature in another module.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/another_module_flat_values_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/another_module_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/another_module_flat_values_float.m %t.float.d/another_module_flat_values_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/another_module_flat_values_float.m %t.float.d/another_module_flat_values_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/another_module_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/another_module_flat_values_float32.m %t.float32.d/another_module_flat_values_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/another_module_flat_values_float32.m %t.float32.d/another_module_flat_values_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 | blue 200 -2 2000 0.5
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 | red 300 -3 3000 0.75
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NOT:{{.}}
:- module another_module_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module another_module_flat_values_lib, int, list, string.

    % The type, make and use are another module's.
:- func run(int) = string.

run(N) = S :-
    Vs = list.map(make, [N, N + 1]),
    S = string.join_list(" | ", list.map(use, Vs)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
