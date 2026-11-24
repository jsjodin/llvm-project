% MANUAL: Exception handling
% The io parameter's state variable must be in scope and of type io.state.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_io_not_state.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    S = 3,
    ( try [io(!S)] true % ERROR
    then
        io.write_int(S, !IO)
    ).
