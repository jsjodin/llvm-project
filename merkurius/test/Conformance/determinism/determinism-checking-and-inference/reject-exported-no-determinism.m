% MANUAL: Determinism checking and inference
% A procedure exported from its module must have a determinism
% annotation.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_exported_no_determinism.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred p(int::in, int::out).                       % ERROR
:- implementation.
p(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
