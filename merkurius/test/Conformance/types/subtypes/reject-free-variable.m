% MANUAL: Subtypes
% A subtype's body's type variables must be among its parameters.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_free_variable.
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

:- type free_var =< list(int)
    --->    [T | list(int)].                            % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
