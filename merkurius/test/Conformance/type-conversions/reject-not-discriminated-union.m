% MANUAL: Type conversions
% Both types must be discriminated union types: int and an equivalence of
% it are not.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_not_discriminated_union.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type count == int.

main(!IO) :-
    X = 3 : int,
    Y = coerce(X) : count,                              % ERROR
    io.write_line(Y, !IO).
