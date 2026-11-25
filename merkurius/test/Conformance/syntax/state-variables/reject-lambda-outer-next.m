% MANUAL: State variables
% A lambda refers to an enclosing state variable only as !.X; the
% manual's example that refers to !:S inside the lambda is illegal.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_lambda_outer_next.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- pred q(int::in, int::out, int::in, int::out) is det.
q(C, C, S, S).
:- pred p(int::in, int::out, int::in, int::out) is det.
p(A, B, !S) :-
    F = ( pred(C::in, D::out) is det :-
            q(C, D, !S) % ERROR
        ),
    F(A, B).
main(!IO) :- io.write_string("ok\n", !IO).
