% MANUAL: Type conversions
% A recursive occurrence of the base type counts as another type unless it
% is of the exact form base(B1, ..., Bn): alt(A, B) recurs as alt(B, A).
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:28:{{[0-9]+}}: error:

:- module reject_recursive_other_form.
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

:- type alt(A, B)
    --->    nil
    ;       cons(A, alt(B, A)).

main(!IO) :-
    X = cons(lemon, nil) : alt(citrus, fruit),
    Y = coerce(X) : alt(fruit, fruit),                  % ERROR
    io.write_line(Y, !IO).
