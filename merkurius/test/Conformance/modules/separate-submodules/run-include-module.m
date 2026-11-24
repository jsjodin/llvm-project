% MANUAL: Separate submodules
% Separate submodules, in files named by their fully qualified names: one
% included in the interface and one in the implementation. Each sees the
% parent's implementation, as a nested one would.
% MODULES: Inputs/run_include_module.shapes.m Inputs/run_include_module.helpers.m
% KNOWN-BUG: a separate submodule does not see its parent's declarations
% (docs/INPROGRESS.md, "Nested submodules").
%
% XFAIL: *
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/run_include_module.shapes.m %S/Inputs/run_include_module.helpers.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/run_include_module.shapes.m %S/Inputs/run_include_module.helpers.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:7
% OUTPUT-NOT:{{.}}

:- module run_include_module.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- include_module shapes.
:- implementation.
:- include_module helpers.
:- import_module list.
:- import_module run_include_module.shapes, run_include_module.helpers.

:- func scale = int.
scale = 3.

main(!IO) :-
    io.write_line(total([square(2), circle(1)]), !IO).
