% MANUAL: Predicate and function mode declarations
% Without a mode declaration, a function's arguments are in and its result
% out, so it may not be called backwards.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_default_function_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- func f(int) = int.
f(X) = X.

:- pred p(int::out) is det.
p(X) :-
    1 = f(X).                                       % ERROR

main(!IO) :-
    p(Y),
    io.write_line(Y, !IO).
