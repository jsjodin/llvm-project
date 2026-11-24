% MANUAL: Abstract solver type declarations
% A solver type that a module exports abstractly, with its definition in
% its implementation section, used by another module.
% MODULES: Inputs/inputs_solver_type.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/inputs_solver_type.m -o %t 2>&1 | FileCheck %s --check-prefix=NYI
% NYI: {{.*}}: error: not yet implemented: code generation for values of type inputs_solver_type.v

:- module nyi_exported.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module inputs_solver_type.

main(!IO) :-
    inputs_solver_type.new(V),
    promise_pure (
        if inputs_solver_type.zero(V) then io.write_string("zero\n", !IO)
        else io.write_string("not zero\n", !IO)
    ).
