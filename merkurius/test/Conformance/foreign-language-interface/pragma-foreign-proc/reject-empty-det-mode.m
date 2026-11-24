% MANUAL: pragma foreign_proc
% A det mode with neither clauses nor a foreign_proc, of a predicate whose
% other mode has a foreign_proc, is inferred failure, which its declaration
% does not allow.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_empty_det_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- pred p(int, int).
:- mode p(in, out) is det.
:- mode p(out, in) is det. % ERROR
:- pragma foreign_proc("C", p(X::in, Y::out),
    [promise_pure, will_not_call_mercury], "Y = X;").

main(!IO) :- p(1, Y), io.write_int(Y, !IO), io.nl(!IO).
