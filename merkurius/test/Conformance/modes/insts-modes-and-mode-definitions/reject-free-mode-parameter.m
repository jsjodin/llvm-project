% MANUAL: Insts modes and mode definitions
% The inst variables of a mode definition's body must be its parameters.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:13:{{[0-9]+}}: error:

:- module reject_free_mode_parameter.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- mode passes(I) == J >> I.                        % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
