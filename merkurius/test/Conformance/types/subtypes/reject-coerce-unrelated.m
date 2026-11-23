% MANUAL: Subtypes
% coerce converts only between a subtype and its supertype, or between
% subtypes of one base type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_coerce_unrelated.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type fruit
    --->    apple
    ;       lemon.

:- type colour
    --->    apple
    ;       lemon.

main(!IO) :-
    F = apple : fruit,
    C = coerce(F) : colour,                             % ERROR
    io.write_line(C, !IO).
