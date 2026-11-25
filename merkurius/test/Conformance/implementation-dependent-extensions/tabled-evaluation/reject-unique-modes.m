% MANUAL: Tabled evaluation
% Tabling does not apply to procedures with unique modes.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_unique_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(int::di, int::uo) is det. % ERROR
:- pragma memo(p/2).
p(X, X).
main(!IO) :- io.write_string("ok\n", !IO).
