% MANUAL: Module initialisation
% An imported module's initialise directive runs its predicate before the
% program's main.
% MODULES: Inputs/init_other.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/init_other.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/init_other.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:init_other initialised
% OUTPUT-NEXT:42
% OUTPUT-NOT:{{.}}

:- module run_imported_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module init_other.

main(!IO) :-
    io.write_line(answer, !IO).
