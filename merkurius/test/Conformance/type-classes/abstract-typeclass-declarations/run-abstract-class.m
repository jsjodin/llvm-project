% MANUAL: Abstract typeclass declarations
% An abstract typeclass declaration in an interface names the class
% without its methods: client modules use it in constraints.
% MODULES: Inputs/shapes.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/shapes.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/shapes.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:18
% OUTPUT-NOT:{{.}}

:- module run_abstract_class.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module shapes.

:- func double_area(T) = int <= shape(T).
double_area(X) = 2 * area(X).

:- import_module int.

main(!IO) :-
    io.write_line(double_area(square(3)), !IO).
