% MANUAL: Subtypes
% A subtype may replace an argument's type only with a subtype of it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_widened_argument.
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

:- type widened =< fruit
    --->    pear(string).                               % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
