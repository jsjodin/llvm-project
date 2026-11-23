% MANUAL: Insts modes and mode definitions
% There must not be more than one mode definition with the same name and
% arity in one module.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:16:{{[0-9]+}}: error:

:- module reject_duplicate_mode.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module list, maybe.

:- mode m1 == in.
:- mode m1 == out.                                  % ERROR

main(!IO) :-
    io.write_string("unreachable\n", !IO).
