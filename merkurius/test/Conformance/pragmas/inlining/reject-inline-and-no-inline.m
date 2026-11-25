% MANUAL: Inlining
% A predicate may not be both inline and no_inline.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_inline_and_no_inline.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred e(int::in, int::out) is det.
:- pragma inline(e/2).
:- pragma no_inline(e/2). % ERROR
e(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
