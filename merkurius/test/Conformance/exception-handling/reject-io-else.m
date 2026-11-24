% MANUAL: Exception handling
% A goal with an io parameter cannot fail, and so has no else part.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_io_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    ( try [io(!IO)] io.write_string("goal\n", !IO) % ERROR
    then
        io.write_string("then\n", !IO)
    else
        io.write_string("else\n", !IO)
    ).
