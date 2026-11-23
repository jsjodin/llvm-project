% MANUAL: Combined higher-order types and insts
% An argument that is only ground, as a predicate's input of a higher-order
% type without an inst, does not have the declared inst.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:20:{{[0-9]+}}: error:

:- module reject_construct_ground.
:- interface.
:- import_module io.

:- type job
    --->    job(pred(int::out, io::di, io::uo) is det).

:- pred wrap(pred(int, io, io)::in, job::out) is det.

:- implementation.

wrap(P, J) :-
    J = job(P).                                     % ERROR
