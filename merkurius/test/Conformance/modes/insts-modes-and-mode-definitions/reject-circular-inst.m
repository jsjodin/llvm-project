% MANUAL: Insts modes and mode definitions
% An inst may not be defined as itself, directly or through others.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_circular_inst.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- inst loop1 == loop2.                             % ERROR
:- inst loop2 == loop1.

main(!IO) :-
    io.write_string("unreachable\n", !IO).
