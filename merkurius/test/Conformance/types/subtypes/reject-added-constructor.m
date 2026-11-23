% MANUAL: Subtypes
% A subtype may not add a constructor its supertype does not have.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:24:{{[0-9]+}}: error:

:- module reject_added_constructor.
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

:- type added =< fruit
    --->    apple
    ;       banana.                                     % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
