% MANUAL: Abstract types
% An abstract type's constructors are not exported: a client cannot use
% them.
% MODULES: Inputs/abstract_stack.m
%
% RUN: not merkc -I %S/Inputs %s %S/Inputs/abstract_stack.m -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_hidden_constructor.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module abstract_stack.

main(!IO) :-
    S = cons(1, nil) : stack(int),                  % ERROR
    io.write_line(depth_to_int(depth(S)), !IO).
