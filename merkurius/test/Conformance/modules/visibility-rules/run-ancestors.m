% MANUAL: Visibility rules
% A submodule of a submodule sees all the declarations of both its
% ancestors, their implementations' too, and the interfaces of the modules
% they import: the leaf calls its grandparent's private function on a
% value of its private type, built from the length of a list, whose module
% only the grandparent imports, and its parent's private function, and
% builds a value of a type the grandparent's interface declares abstractly.
% The leaf's interface names the grandparent's private type, which the
% grandparent uses through it.
% MODULES: Inputs/vis_family.m Inputs/vis_family.mid.m Inputs/vis_family.mid.leaf.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/vis_family.m %S/Inputs/vis_family.mid.m %S/Inputs/vis_family.mid.leaf.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/vis_family.m %S/Inputs/vis_family.mid.m %S/Inputs/vis_family.mid.leaf.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:18
% OUTPUT-NEXT:1
% OUTPUT-NEXT:5
% OUTPUT-NOT:{{.}}

:- module run_ancestors.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module vis_family.

main(!IO) :-
    vis_family.run(!IO).
