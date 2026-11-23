% MANUAL: Abstract types
% The implementation section must define every abstract type its interface
% names.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:12:{{[0-9]+}}: error:

:- module reject_undefined.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- type secret.                                     % ERROR
:- implementation.
main(!IO) :-
    io.write_string("unreachable\n", !IO).
