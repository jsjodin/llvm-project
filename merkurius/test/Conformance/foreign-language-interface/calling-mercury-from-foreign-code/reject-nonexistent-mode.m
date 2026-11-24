% MANUAL: Calling Mercury from foreign code
% A foreign_export must name a mode of its predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_nonexistent_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::in, int::out) is det.
p(X, X).
:- pragma foreign_export("C", p(out, in), "p_from_c"). % ERROR

main(!IO) :- p(1, Y), io.write_int(Y, !IO), io.nl(!IO).
