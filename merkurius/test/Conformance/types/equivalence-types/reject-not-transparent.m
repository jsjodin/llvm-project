% MANUAL: Equivalence types
% An equivalence type must be transparent: a type variable on its right must
% occur on its left.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_not_transparent.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type anything == list(T).                        % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
