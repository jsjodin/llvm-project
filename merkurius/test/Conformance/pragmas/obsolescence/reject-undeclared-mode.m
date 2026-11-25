% MANUAL: Obsolescence
% An obsolete_proc pragma names a mode the predicate has.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_undeclared_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(int::in, int::out) is det.
:- pragma obsolete_proc(e(out, in)). % ERROR
e(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
