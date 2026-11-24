% MANUAL: Type conversions
% Corresponding type arguments must be equal, or one a subtype of the other:
% two subtypes of fruit with no relation between them are neither.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:29:{{[0-9]+}}: error:

:- module reject_unrelated_arguments.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type fruit
    --->    apple
    ;       lemon
    ;       orange.

:- type citrus =< fruit
    --->    lemon
    ;       orange.

:- type pome =< fruit
    --->    apple.

main(!IO) :-
    X = [lemon] : list(citrus),
    Y = coerce(X) : list(pome),                         % ERROR
    io.write_line(Y, !IO).
