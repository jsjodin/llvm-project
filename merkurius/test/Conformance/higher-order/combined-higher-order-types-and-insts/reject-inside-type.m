% MANUAL: Combined higher-order types and insts
% A higher-order type with modes and a determinism is allowed only as a
% direct argument of a constructor: not inside another type's argument.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_inside_type.
:- interface.
:- import_module io, list.

:- type jobs
    --->    jobs(list(pred(int::out, io::di, io::uo) is det)). % ERROR
