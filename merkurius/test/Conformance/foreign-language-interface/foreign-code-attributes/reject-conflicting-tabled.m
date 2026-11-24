% MANUAL: Foreign code attributes
% may_call_mm_tabled needs may_call_mercury.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_conflicting_tabled.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::out) is det.
:- pragma foreign_proc("C", p(X::out),
    [promise_pure, will_not_call_mercury, may_call_mm_tabled], "X = 1;"). % ERROR

main(!IO) :- p(X), io.write_int(X, !IO), io.nl(!IO).
