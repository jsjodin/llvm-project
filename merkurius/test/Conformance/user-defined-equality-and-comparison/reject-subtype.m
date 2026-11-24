% MANUAL: User-defined equality and comparison
% A subtype may not have a user-defined equality.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_subtype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type fruit
    --->    apple
    ;       orange
    ;       lemon.

:- type citrus =< fruit                             % ERROR
    --->    orange
    ;       lemon
            where equality is citrus_equals.

:- pred citrus_equals(citrus::in, citrus::in) is semidet.
citrus_equals(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
