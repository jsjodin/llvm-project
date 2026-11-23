% MANUAL: Subtypes
% A subtype must not have user-defined equality.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:22:{{[0-9]+}}: error:

:- module reject_user_equality.
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

:- type with_equality =< fruit                          % ERROR
    --->    apple
    where equality is eq.

:- pred eq(with_equality::in, with_equality::in) is semidet.
eq(_, _).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
