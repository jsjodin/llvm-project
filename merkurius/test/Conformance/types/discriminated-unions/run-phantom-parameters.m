% MANUAL: Discriminated unions
% A type parameter need not occur in the body, a phantom parameter: the
% manual's var(T), with prog_var and type_var binding different types to T,
% share a representation and are distinct types.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1
% OUTPUT-NEXT:2
% OUTPUT-NEXT:var(1)
% OUTPUT-NOT:{{.}}

:- module run_phantom_parameters.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type var(T)
    --->    var(int).

:- type prog_var == var(prog_var_type).
:- type type_var == var(type_var_type).

:- type prog_var_type
    --->    prog_var_type.
:- type type_var_type
    --->    type_var_type.

main(!IO) :-
    P = var(1) : prog_var,
    T = var(2) : type_var,
    io.write_line(number(P), !IO),
    io.write_line(number(T), !IO),
    io.write_line(P, !IO).

:- func number(var(T)) = int.

number(var(N)) = N.
