% MANUAL: Tabled evaluation
% disable_warning_if_ignored may not be given for loop_check, which
% changes what a program means.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_disable_warning_loop_check.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is det.
:- pragma loop_check(p/2, [disable_warning_if_ignored]). % ERROR
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
