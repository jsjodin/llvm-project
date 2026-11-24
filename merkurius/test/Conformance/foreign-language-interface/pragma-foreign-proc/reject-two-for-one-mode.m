% MANUAL: pragma foreign_proc
% A mode may have at most one foreign_proc for each language.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_two_for_one_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::in, int::out) is det.
:- pragma foreign_proc("C", p(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X;").
:- pragma foreign_proc("C", p(X::in, Y::out), % ERROR
    [promise_pure, will_not_call_mercury], "Y = X + 1;").

main(!IO) :- p(1, Y), io.write_int(Y, !IO), io.nl(!IO).
