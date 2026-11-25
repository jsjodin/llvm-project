% MANUAL: Goals
% require_det of a goal that is not det is an error.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_require_det.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred p(list(int)::in, int::out) is nondet.
p(L, X) :-
    require_det ( list.member(X, L), X > 1 ). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
