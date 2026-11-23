% MANUAL: Predicate and function mode declarations
% The mode declarations of a predicate declared in the interface must be
% in the interface too.
%
% RUN: not merkc %s -o %t 2>&1 | FileCheck %s --check-prefix=ERRORS --implicit-check-not=error:
% ERRORS-DAG: {{.*}}:14:{{[0-9]+}}: error:

:- module reject_mode_in_other_section.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- pred p(int, int).
:- implementation.
:- mode p(in, out) is det.                          % ERROR
p(X, X).

main(!IO) :-
    io.write_string("unreachable\n", !IO).
