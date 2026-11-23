% MANUAL: Insts modes and mode definitions
% A user-defined mode may not have a reserved mode name. mmc reserves
% only >> and is of the manual's list, which merkc follows.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:15:{{[0-9]+}}: error:

:- module reject_reserved_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- mode (is) == in.                                 % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
