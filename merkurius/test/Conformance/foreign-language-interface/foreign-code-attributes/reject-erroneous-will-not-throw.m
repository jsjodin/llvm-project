% MANUAL: Foreign code attributes
% An erroneous procedure cannot return, so it must throw: its foreign_proc
% may not promise will_not_throw_exception.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_erroneous_will_not_throw.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int::out) is erroneous. % ERROR
:- pragma foreign_proc("C", p(X::out),
    [promise_pure, will_not_call_mercury, will_not_throw_exception],
    "X = 1; abort();").

main(!IO) :- p(X), io.write_int(X, !IO), io.nl(!IO).
