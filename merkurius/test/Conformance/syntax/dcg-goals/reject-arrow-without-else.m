% MANUAL: DCG-goals
% A DCG if-then-else written with -> has its ; part.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_arrow_without_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(list(int)::in, list(int)::out) is semidet.
p --> ( [1] -> [2] ). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
