% MANUAL: Abstract instance declarations
% The manual's hashable module: its abstract instances in the interface
% let a client module call hash at int and string.
% MODULES: Inputs/hashable.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/hashable.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/hashable.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NEXT:same hash
% OUTPUT-NOT:{{.}}

:- module run_hashable.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module hashable.

main(!IO) :-
    io.write_line(hash(42), !IO),
    ( if hash("abc") = hash("abc") then
        io.write_string("same hash\n", !IO)
    else
        io.write_string("different hashes\n", !IO)
    ).
