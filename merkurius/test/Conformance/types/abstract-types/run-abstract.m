% MANUAL: Abstract types
% A type declared without a definition in an interface is abstract: client
% modules use its name and the operations its module exports, which defines
% it in its implementation as a discriminated union or an equivalence type.
% MODULES: Inputs/abstract_stack.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/abstract_stack.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/abstract_stack.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:3
% OUTPUT-NEXT:3
% OUTPUT-NEXT:2
% OUTPUT-NOT:{{.}}

:- module run_abstract.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module abstract_stack.

main(!IO) :-
    S = push(3, push(2, push(1, empty))),
    io.write_line(depth_to_int(depth(S)), !IO),
    ( if pop(S, X, S1) then
        io.write_line(X, !IO),
        io.write_line(depth_to_int(depth(S1)), !IO)
    else
        io.write_string("empty\n", !IO)
    ).
