% MANUAL: Insts modes and mode definitions
% A mode mapping may not take a node of the type tree from bound to free.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_bound_to_free.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.

:- mode unbind == ground >> free.

:- pred forget(int::unbind) is det.
forget(_).                                          % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
