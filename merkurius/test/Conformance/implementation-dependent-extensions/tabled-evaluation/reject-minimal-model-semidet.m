% MANUAL: Tabled evaluation
% Minimal model tabling is for procedures that can succeed more than once.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_minimal_model_semidet.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::in, int::out) is semidet. % ERROR
:- pragma minimal_model(p/2).
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
