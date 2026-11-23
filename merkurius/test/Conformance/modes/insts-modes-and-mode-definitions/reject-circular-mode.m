% MANUAL: Insts modes and mode definitions
% A mode may not be defined as itself, directly or through others.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_circular_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- mode loop1 == loop2.                             % ERROR
:- mode loop2 == loop1.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
