% MANUAL: State variables
% !X may not be an argument of a function application.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_function_argument.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.
:- func f(int, int) = int.
f(X, Y) = X + Y.
:- pred p(int::in, int::out) is det.
p(!S) :-
    _ = f(!S). % ERROR
main(!IO) :- io.write_string("ok\n", !IO).
