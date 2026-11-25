% MANUAL: Field access functions
% MANUAL: Modules
% Interaction: field access and update functions, in the feature in another
% module.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/another_module_field_access_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/another_module_field_access_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/another_module_field_access_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: owner1 15 10 | owner2 25 20
% OUTPUT-NEXT:2: owner2 25 20 | owner3 35 30
% OUTPUT-NEXT:3: owner3 35 30 | owner4 45 40
% OUTPUT-NEXT:4: owner4 45 40 | owner5 55 50
% OUTPUT-NOT:{{.}}
:- module another_module_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module another_module_field_access_lib, int, list, string.

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
