% MANUAL: Equivalence types
% An equivalence type must not be cyclic: its name may not occur on its
% right side, directly or through another equivalence.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:18:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_cyclic.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list.

:- type loop == list(loop).                         % ERROR
:- type ping == pong.                               % ERROR
:- type pong == ping.                               % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
