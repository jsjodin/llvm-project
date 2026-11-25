% MANUAL: The module system
% Two modules that import each other: each one's compile finds itself in
% the other's interface, where the other names its type, and that is the
% type it defines, not a copy of it read from its source again.
% MODULES: Inputs/mutual_colours.m Inputs/mutual_names.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/mutual_colours.m %S/Inputs/mutual_names.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/mutual_colours.m %S/Inputs/mutual_names.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:red
% OUTPUT-NEXT:green
% OUTPUT-NOT:{{.}}

:- module run_mutual_imports.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module mutual_colours.

main(!IO) :-
    io.write_string(name(red), !IO),
    io.nl(!IO),
    io.write_string(name(green), !IO),
    io.nl(!IO).
