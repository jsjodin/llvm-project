% MANUAL: Exception handling
% The variables GOAL binds are not bound in a catch.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:19:{{[0-9]+}}: error:

:- module reject_catch_sees_goal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] X = 1
    then
        io.write_int(X, !IO)
    catch "a" ->
        io.write_int(X, !IO) % ERROR
    ).
