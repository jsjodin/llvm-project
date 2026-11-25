% DESIGN: Loops / The option
% Without -fdo-loops there is no do operator, so a file reads exactly as
% mmc reads it, and a loop is a syntax error where mmc reports one.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:
:- module reject_without_option.
:- interface.
:- import_module list.
:- pred sum(list(int)::in, int::in, int::out) is det.
:- implementation.
:- import_module int.
sum(Xs, !S) :-
    ( foreach(X, Xs) do !:S = !.S + X ).                % ERROR
