% MANUAL: Combined higher-order types and insts
% A term may be constructed only with an argument that has the inst its
% type declares: here the manual's p, whose first argument is in, not out.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_construct_other_inst.
:- interface.
:- import_module io.

:- type job
    --->    job(pred(int::out, io::di, io::uo) is det).

:- pred bad(job::out) is det.

:- implementation.

bad(job(p)).                                        % ERROR

:- pred p(int::in, io::di, io::uo) is det.
p(_, !IO).
