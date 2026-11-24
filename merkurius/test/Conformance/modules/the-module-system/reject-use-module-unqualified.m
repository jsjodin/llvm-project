% MANUAL: The module system
% A name imported with use_module must be fully module qualified.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_use_module_unqualified.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- use_module string.

main(!IO) :-
    N = length("abc"),                              % ERROR
    io.write_int(N, !IO).
