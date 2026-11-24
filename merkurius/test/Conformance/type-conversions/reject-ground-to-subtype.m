% MANUAL: Type conversions
% The manual's f2: a ground fruit cannot be converted to citrus, since some
% fruits are not citrus.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:23:{{[0-9]+}}: error:

:- module reject_ground_to_subtype.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- func f2(fruit) = citrus.
f2(X) = coerce(X).                                      % ERROR

main(!IO) :-
    io.write_line(f2(lemon), !IO).
