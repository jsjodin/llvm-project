% MANUAL: Calling higher-order terms
% A function term is applied with apply, not called with call.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_call_function.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

main(!IO) :-
    F = (func(X) = X + 1),
    call(F, 1, Y),                                  % ERROR
    io.write_line(Y, !IO).
