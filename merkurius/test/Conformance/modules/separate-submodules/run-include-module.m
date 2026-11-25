% MANUAL: Separate submodules
% Separate submodules, in files named by their fully qualified names: one
% included in the interface and one in the implementation. Each sees the
% parent's implementation, as a nested one would, and the one with
% end_module names itself there by its last component. The program's main
% module imports the parent.
% MODULES: Inputs/include_parent.m Inputs/include_parent.shapes.m Inputs/include_parent.helpers.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/include_parent.m %S/Inputs/include_parent.shapes.m %S/Inputs/include_parent.helpers.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/include_parent.m %S/Inputs/include_parent.shapes.m %S/Inputs/include_parent.helpers.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:7
% OUTPUT-NOT:{{.}}

:- module run_include_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module include_parent.

main(!IO) :-
    include_parent.run(!IO).
