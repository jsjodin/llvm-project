% MANUAL: Type conversions
% A bound inst's arguments are checked in turn: a pair whose first argument
% is bound to lemon but whose second is any ground fruit does not convert to
% pair(citrus, citrus).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_ground_argument.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module pair.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- pred to_citrus(pair(fruit, fruit)::in(bound(bound(lemon) - ground)),
    pair(citrus, citrus)::out) is det.
to_citrus(P, Q) :-
    Q = coerce(P).                                      % ERROR

main(!IO) :-
    to_citrus(lemon - orange, Q),
    io.write_line(Q, !IO).
