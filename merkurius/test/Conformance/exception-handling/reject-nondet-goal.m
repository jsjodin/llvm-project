% MANUAL: Exception handling
% GOAL must be det, semidet, cc_multi or cc_nondet: a nondet goal is not,
% though here an else part makes the failure possible.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:17:{{[0-9]+}}: error:

:- module reject_nondet_goal.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi. % ERROR
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] list.member(X, [1, 2, 3]) % ERROR
    then
        io.write_int(X, !IO)
    ).
