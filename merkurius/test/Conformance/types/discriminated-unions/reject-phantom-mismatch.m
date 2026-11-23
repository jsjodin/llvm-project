% MANUAL: Discriminated unions
% Values whose phantom parameters are bound to different types have
% different types, and cannot be unified.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_phantom_mismatch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type var(T)
    --->    var(int).

:- type prog_var_type
    --->    prog_var_type.
:- type type_var_type
    --->    type_var_type.

main(!IO) :-
    P = var(1) : var(prog_var_type),
    T = var(1) : var(type_var_type),
    ( if P = T then                                 % ERROR
        io.write_string("same\n", !IO)
    else
        io.write_string("different\n", !IO)
    ).
