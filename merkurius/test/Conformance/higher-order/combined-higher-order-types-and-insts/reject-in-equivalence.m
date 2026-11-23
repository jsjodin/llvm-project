% MANUAL: Combined higher-order types and insts
% A higher-order type with modes and a determinism is allowed only as a
% direct argument of a constructor: not on the right of an equivalence.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_in_equivalence.
:- interface.
:- import_module io.

:- type p == (pred(io::di, io::uo) is det).         % ERROR
:- type f == (func(int::in) = (int::out) is semidet). % ERROR
