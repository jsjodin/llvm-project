% MANUAL: Exception handling
% CATCHANYVAR is bound only in CATCHANYGOAL: where GOAL can succeed, the
% goals after the try goal cannot use it.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_catch_any_var_after.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    % On one line, as mmc reports the error at the catch's last goal.
    ( try [] X = 1 then io.write_int(X, !IO) catch_any E -> true ), % ERROR
    io.write(E, !IO).
