% MANUAL: pragma foreign_proc
% A foreign_proc must be for a declared mode of its predicate.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_undeclared_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::in, int::out) is det.
p(X, X).
:- pragma foreign_proc("C", p(X::out, Y::in), % ERROR
    [promise_pure, will_not_call_mercury], "X = Y;").

main(!IO) :- p(1, Y), io.write_int(Y, !IO), io.nl(!IO).
