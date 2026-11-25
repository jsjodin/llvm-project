% MANUAL: Unique modes
% MANUAL: Modules
% Interaction: an array updated in place in unique modes, in the feature in
% another module.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/another_module_unique_modes_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/another_module_unique_modes_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/another_module_unique_modes_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 0 1 4 9 | 0 1 4 9 16
% OUTPUT-NEXT:2: 0 1 4 9 16 | 0 1 4 9 16 25
% OUTPUT-NEXT:3: 0 1 4 9 16 25 | 0 1 4 9 16 25 36
% OUTPUT-NEXT:4: 0 1 4 9 16 25 36 | 0 1 4 9 16 25 36 49
% OUTPUT-NOT:{{.}}
:- module another_module_unique_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module another_module_unique_modes_lib, int, list, string.

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
