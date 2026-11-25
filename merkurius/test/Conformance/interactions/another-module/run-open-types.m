% DESIGN: Open terms (Prolog-style variables)
% MANUAL: Modules
% Interaction: a list built as a difference list and taken apart by unifying
% open terms, in the feature in another module.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fopen-types
% TRANSLATION: Inputs/another_module_open_types_translation.m
% MODULES: Inputs/another_module_open_types_lib.m
% FILES: Inputs/another_module_open_types_lib_translation.m
%
% RUN: merkc -O0 -fopen-types -I %S/Inputs %s %S/Inputs/another_module_open_types_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fopen-types -I %S/Inputs %s %S/Inputs/another_module_open_types_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 1 10 rest  | 1 2 rest 20
% OUTPUT-NEXT:2: 1 2 rest 20 | 1 2 rest 3,30
% OUTPUT-NEXT:3: 1 2 rest 3,30 | 1 2 rest 3,4,40
% OUTPUT-NEXT:4: 1 2 rest 3,4,40 | 1 2 rest 3,4,5,50
% OUTPUT-NOT:{{.}}
:- module another_module_open_types.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module another_module_open_types_lib, int, list, string.

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
