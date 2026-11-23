% MANUAL: Discriminated unions
% A user-defined type may not have a reserved name, such as int or string.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_reserved_name.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- type int                                         % ERROR
    --->    i.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
