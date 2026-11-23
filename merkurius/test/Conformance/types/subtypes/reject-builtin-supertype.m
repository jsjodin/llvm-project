% MANUAL: Subtypes
% A subtype's supertype must be a discriminated union type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_builtin_supertype.
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

:- type of_int =< int                                   % ERROR
    --->    zero.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
