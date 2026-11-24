% MANUAL: Type conversions
% The manual's f4: the parameter T of wrap/1 is fruit in the input type and
% citrus in the result type, but it must be the same where the types have it
% in a higher-order type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_higher_order_parameter.
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

:- type wrap(T)
    --->    wrap(T).

:- func f4(func(fruit) = int) = (func(citrus) = int).
f4(X) = Y :-
    wrap(Y) = coerce(wrap(X)).                          % ERROR

main(!IO) :-
    io.write_line(f4(func(_) = 1)(lemon), !IO).
