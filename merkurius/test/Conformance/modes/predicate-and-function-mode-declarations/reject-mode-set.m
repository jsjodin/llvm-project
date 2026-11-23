% MANUAL: Predicate and function mode declarations
% A call must satisfy the initial insts of one of its callee's modes, or
% one they imply.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_mode_set.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred q(int::in, int::out) is det.
q(X, X).

:- pred p(int::out) is det.
p(Y) :-
    q(_, Y).                                        % ERROR

main(!IO) :-
    p(Y),
    io.write_line(Y, !IO).
