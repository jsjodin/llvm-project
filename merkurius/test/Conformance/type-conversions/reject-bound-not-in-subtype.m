% MANUAL: Type conversions
% Each function symbol of a bound inst must name a constructor of the result
% type: an inst of lemon or apple does not fit citrus.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_bound_not_in_subtype.
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

:- inst lemon_or_apple for fruit/0
    --->    lemon
    ;       apple.

:- func f(fruit) = citrus.
:- mode f(in(lemon_or_apple)) = out is det.
f(X) = coerce(X).                                       % ERROR

main(!IO) :-
    io.write_line(f(lemon), !IO).
