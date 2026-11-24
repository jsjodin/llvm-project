% MANUAL: Type conversions
% Higher-order types are not discriminated union types, so a closure cannot
% be coerced, even to its own type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_higher_order_type.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
main(!IO) :-
    F = (func(X) = X + 1) : (func(int) = int),
    G = coerce(F) : (func(int) = int),                  % ERROR
    io.write_line(G(1), !IO).

:- import_module int.
