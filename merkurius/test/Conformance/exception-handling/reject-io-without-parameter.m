% MANUAL: Exception handling
% If no io parameter exists, GOAL may not perform I/O.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_io_without_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [] io.write_string("goal\n", !IO) % ERROR
    then
        io.write_string("then\n", !IO)
    ).
