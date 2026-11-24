% MANUAL: Exception handling
% The else part is mandatory if GOAL may fail.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:11:{{[0-9]+}}: error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_semidet_without_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi. % ERROR
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] ( X = 1, X > 0 ) % ERROR
    then
        io.write_string("then\n", !IO)
    ).
