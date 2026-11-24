% MANUAL: Exception handling
% A catch that succeeds without binding what GOAL binds leaves those
% variables unbound after the try goal.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_goal_bindings_after_catch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    % On one line, as mmc reports the error at the catch's last goal.
    ( try [] X = 1 then true catch_any _ -> true ), % ERROR
    io.write_int(X, !IO).
