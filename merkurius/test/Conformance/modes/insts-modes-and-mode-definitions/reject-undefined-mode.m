% MANUAL: Insts modes and mode definitions
% A mode must be defined before it is used.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_undefined_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- pred p(int::nowhere) is det.                     % ERROR
p(_).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
