% MANUAL: Insts modes and mode definitions
% A user-defined inst may not have a reserved inst name.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_reserved_inst.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- inst clobbered == ground.                        % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
