% MANUAL: Subtypes
% coerce converts from a subtype to its supertype, not the other way.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_coerce_down.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type fruit
    --->    apple
    ;       lemon
    ;       pear(int).

:- type citrus =< fruit
    --->    lemon.

main(!IO) :-
    F = apple : fruit,
    C = coerce(F) : citrus,                             % ERROR
    io.write_line(C, !IO).
