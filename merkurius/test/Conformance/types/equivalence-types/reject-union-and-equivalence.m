% MANUAL: Equivalence types
% A type may not be defined both as a discriminated union and as an
% equivalence type.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_union_and_equivalence.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- type colour == int.
:- type colour                                      % ERROR
    --->    red.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
